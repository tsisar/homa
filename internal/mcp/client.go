// Package mcp connects to an MCP server (or a federating gateway like
// kgateway/agentgateway) over Streamable HTTP, discovers its tools, and
// exposes an allow-listed subset to the agent's tool-calling loop.
package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"

	"agent/internal/lemonade"

	"github.com/tsisar/extended-log-go/log"
)

// Executor is a single Streamable-HTTP MCP connection. Because the kgateway
// MCP gateway already federates multiple backends behind one endpoint, a
// single client sees the union of all tools — no multi-server manager needed.
type Executor struct {
	url   string
	allow []string

	mu     sync.Mutex
	client *mcpclient.Client
	tools  []lemonade.Tool // allow-listed
	ds     *dsCache        // Grafana datasource name/type -> UID resolver (nil if no grafana tools)
}

// New connects to the MCP endpoint, lists tools, and keeps only those matching
// the allow list. An empty allow list exposes NO tools (safe default).
func New(ctx context.Context, url string, allow []string) (*Executor, error) {
	e := &Executor{url: url, allow: allow}
	if err := e.connect(ctx); err != nil {
		return nil, err
	}

	res, err := e.client.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		_ = e.Close()
		return nil, fmt.Errorf("list tools: %w", err)
	}
	for _, t := range res.Tools {
		if matchAllow(t.Name, allow) {
			e.tools = append(e.tools, convertTool(t))
		}
	}
	// Small models routinely pass a datasource name/type (or a typo'd UID) where
	// Grafana wants the opaque UID. Prefetch the datasource list once so we can
	// rewrite those arguments. Fail-open: any error just disables resolution.
	if anyGrafana(e.tools) {
		if e.ds = buildDSCache(ctx, e.client); e.ds != nil {
			log.Printf("grafana: cached %d datasource(s) for UID resolution", len(e.ds.list))
		}
	}
	return e, nil
}

func (e *Executor) connect(ctx context.Context) error {
	c, err := mcpclient.NewStreamableHttpClient(e.url)
	if err != nil {
		return fmt.Errorf("create MCP client: %w", err)
	}
	if err := c.Start(ctx); err != nil {
		return fmt.Errorf("start MCP client: %w", err)
	}
	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{Name: "voice-agent", Version: "0.1.0"}
	if _, err := c.Initialize(ctx, initReq); err != nil {
		return fmt.Errorf("initialize MCP session: %w", err)
	}
	e.client = c
	return nil
}

// Tools returns the allow-listed tools to advertise to the LLM.
func (e *Executor) Tools() []lemonade.Tool { return e.tools }

// CallTool invokes a tool by name with raw-JSON arguments and returns its text
// plus any images it produced (e.g. a rendered Grafana panel).
func (e *Executor) CallTool(ctx context.Context, name, argsJSON string) (string, []lemonade.Image, error) {
	var args map[string]any
	if s := strings.TrimSpace(argsJSON); s != "" {
		if err := json.Unmarshal([]byte(s), &args); err != nil {
			return "", nil, fmt.Errorf("bad tool arguments: %w", err)
		}
	}
	e.ds.fix(name, args) // resolve a friendly Grafana datasource name/type to its UID

	e.mu.Lock()
	defer e.mu.Unlock()

	req := mcp.CallToolRequest{}
	req.Params.Name = name
	req.Params.Arguments = args

	res, err := e.client.CallTool(ctx, req)
	if err != nil {
		return "", nil, fmt.Errorf("call %q: %w", name, err)
	}
	text, images := extractContent(res)
	if res.IsError {
		return "", nil, fmt.Errorf("tool %q reported error: %s", name, text)
	}
	return text, images, nil
}

func (e *Executor) Close() error {
	if e.client != nil {
		return e.client.Close()
	}
	return nil
}

func convertTool(t mcp.Tool) lemonade.Tool {
	params := map[string]any{"type": "object"}
	if t.InputSchema.Properties != nil {
		params["properties"] = t.InputSchema.Properties
	}
	if len(t.InputSchema.Required) > 0 {
		params["required"] = t.InputSchema.Required
	}
	return lemonade.Tool{Name: t.Name, Description: t.Description, Parameters: params}
}

func extractContent(res *mcp.CallToolResult) (string, []lemonade.Image) {
	var b strings.Builder
	var images []lemonade.Image
	for _, c := range res.Content {
		if tc, ok := c.(mcp.TextContent); ok {
			if b.Len() > 0 {
				b.WriteString("\n")
			}
			b.WriteString(tc.Text)
			continue
		}
		if ic, ok := mcp.AsImageContent(c); ok {
			images = append(images, lemonade.Image{MIME: ic.MIMEType, Data: ic.Data})
		}
	}
	return b.String(), images
}

// ---- Grafana datasource resolution --------------------------------------

type dsInfo struct {
	uid, name, dtype string
	isDefault        bool
}

// dsCache maps a Grafana datasource's lowercased name, type, or UID to its real
// UID, so a model that passes "prometheus" (or a mistyped UID) still hits the
// right datasource.
type dsCache struct {
	byKey map[string]string // lowercased name/type/uid -> real uid
	def   string            // default datasource uid
	list  []dsInfo
}

func anyGrafana(tools []lemonade.Tool) bool {
	for _, t := range tools {
		if strings.HasPrefix(t.Name, "grafana_") {
			return true
		}
	}
	return false
}

// buildDSCache fetches the datasource list (the tool name is the same whether or
// not it is allow-listed — the allow list only gates what the model sees). Any
// failure returns nil, disabling resolution.
func buildDSCache(ctx context.Context, c *mcpclient.Client) *dsCache {
	req := mcp.CallToolRequest{}
	req.Params.Name = "grafana_list_datasources"
	req.Params.Arguments = map[string]any{}
	res, err := c.CallTool(ctx, req)
	if err != nil || res == nil || res.IsError {
		return nil
	}
	text, _ := extractContent(res)
	return parseDatasources(text)
}

// parseDatasources builds a dsCache from a grafana_list_datasources response
// ({"datasources":[{uid,name,type,isDefault}]}). A key that several datasources
// share is resolved only if they all point to the same UID, or exactly one is
// the default; otherwise it is left ambiguous (skipped) rather than guessed.
func parseDatasources(text string) *dsCache {
	var payload struct {
		Datasources []struct {
			UID       string `json:"uid"`
			Name      string `json:"name"`
			Type      string `json:"type"`
			IsDefault bool   `json:"isDefault"`
		} `json:"datasources"`
	}
	if json.Unmarshal([]byte(text), &payload) != nil || len(payload.Datasources) == 0 {
		return nil
	}

	type cand struct {
		uid string
		def bool
	}
	cands := map[string][]cand{}
	add := func(key, uid string, def bool) {
		if key = strings.ToLower(strings.TrimSpace(key)); key != "" {
			cands[key] = append(cands[key], cand{uid, def})
		}
	}

	dc := &dsCache{byKey: map[string]string{}}
	for _, d := range payload.Datasources {
		if d.UID == "" {
			continue
		}
		dc.list = append(dc.list, dsInfo{d.UID, d.Name, d.Type, d.IsDefault})
		if d.IsDefault {
			dc.def = d.UID
		}
		add(d.UID, d.UID, d.IsDefault) // identity, so a correct UID passes through
		add(d.Name, d.UID, d.IsDefault)
		add(d.Type, d.UID, d.IsDefault)
	}

	for key, list := range cands {
		uniq := map[string]bool{}
		for _, c := range list {
			uniq[c.uid] = true
		}
		uid := ""
		if len(uniq) == 1 {
			uid = list[0].uid // all candidates agree
		} else {
			var defs []string
			for _, c := range list {
				if c.def {
					defs = append(defs, c.uid)
				}
			}
			if len(defs) == 1 {
				uid = defs[0] // a unique default breaks the tie
			}
		}
		if uid != "" {
			dc.byKey[key] = uid
		}
	}
	return dc
}

// fix rewrites a Grafana datasource argument in place when it is a friendly
// name/type/"default" (or wrong case). It touches only datasourceUid /
// data_source_uid — never the generic "uid" key, which other tools use for
// dashboard/incident ids. nil receiver and non-grafana tools are no-ops.
func (dc *dsCache) fix(name string, args map[string]any) {
	if dc == nil || !strings.HasPrefix(name, "grafana_") {
		return
	}
	for _, k := range []string{"datasourceUid", "data_source_uid"} {
		v, ok := args[k].(string)
		if !ok || v == "" {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(v))
		real := dc.byKey[key]
		if real == "" && key == "default" {
			real = dc.def
		}
		if real != "" && real != v {
			log.Printf("grafana: resolved %s %q -> %q for %s", k, v, real, name)
			args[k] = real
		}
		return // at most one datasource key per call
	}
}

// DatasourceHint renders a compact datasource block for the system prompt so the
// model passes friendly names (which fix resolves) instead of guessing UIDs. It
// returns "" when there are no Grafana datasources.
func (e *Executor) DatasourceHint() string {
	if e.ds == nil || len(e.ds.list) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Available Grafana datasources — pass the name as the datasourceUid argument (it is resolved to the real UID automatically); never invent or guess a UID:\n")
	for _, d := range e.ds.list {
		b.WriteString("- ")
		b.WriteString(d.name)
		b.WriteString(" (")
		b.WriteString(d.dtype)
		if d.isDefault {
			b.WriteString(", default")
		}
		b.WriteString(")\n")
	}
	b.WriteString("Before querying a metric you are unsure of, call the matching list tool (e.g. grafana_list_prometheus_metric_names) once and copy the exact name; do not guess metric names.")
	return b.String()
}

// matchAllow reports whether a tool name is permitted. Supports exact names,
// "*" (everything), and a trailing "*" prefix glob (e.g. "ddg_*").
func matchAllow(name string, allow []string) bool {
	for _, p := range allow {
		p = strings.TrimSpace(p)
		switch {
		case p == "":
			continue
		case p == "*":
			return true
		case strings.HasSuffix(p, "*"):
			if strings.HasPrefix(name, strings.TrimSuffix(p, "*")) {
				return true
			}
		case p == name:
			return true
		}
	}
	return false
}
