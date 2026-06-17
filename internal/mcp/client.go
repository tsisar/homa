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
	"time"

	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"

	"agent/internal/lemonade"

	"github.com/tsisar/extended-log-go/log"
)

// Reconnect modes select how a dropped MCP session is recovered (MCP_RECONNECT).
const (
	ReconnectAuto = "auto" // silently reconnect with backoff on a transport failure
	ReconnectAsk  = "ask"  // expose reconnect_tools so the model asks the user first
	ReconnectOff  = "off"  // never reconnect; a failed call just errors

	reconnectToolName = "reconnect_tools"
)

// Executor is a single Streamable-HTTP MCP connection. Because the kgateway
// MCP gateway already federates multiple backends behind one endpoint, a
// single client sees the union of all tools — no multi-server manager needed.
type Executor struct {
	url   string
	allow []string
	mode  string // reconnect policy: ReconnectAuto | ReconnectAsk | ReconnectOff

	mu     sync.Mutex
	client *mcpclient.Client
	tools  []lemonade.Tool // allow-listed (+ reconnect_tools in ask mode)
	ds     *dsCache        // Grafana datasource name/type -> UID resolver (nil if no grafana tools)

	lastDial  time.Time // last (re)connect attempt; gates the reconnect rate
	dialFails int       // consecutive reconnect failures; grows the backoff
}

// New connects to the MCP endpoint, lists tools, and keeps only those matching
// the allow list. An empty allow list exposes NO tools (safe default).
func New(ctx context.Context, url string, allow []string, reconnect string) (*Executor, error) {
	e := &Executor{url: url, allow: allow, mode: normalizeMode(reconnect)}
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
	// In ask mode the model recovers a dropped session by calling this built-in
	// tool (after asking the user); it bypasses the MCP_ALLOW list on purpose.
	if e.mode == ReconnectAsk {
		e.tools = append(e.tools, reconnectToolDef())
	}
	return e, nil
}

// normalizeMode keeps the reconnect policy to a known value; anything unset or
// unrecognized falls back to auto (the prior behavior), never to a broken mode.
func normalizeMode(m string) string {
	switch strings.ToLower(strings.TrimSpace(m)) {
	case ReconnectAsk:
		return ReconnectAsk
	case ReconnectOff:
		return ReconnectOff
	default:
		return ReconnectAuto
	}
}

func reconnectToolDef() lemonade.Tool {
	return lemonade.Tool{
		Name:        reconnectToolName,
		Description: "Reconnect to the tool backend after it has dropped (tool calls were failing). Only call this once the user has agreed to reconnect.",
		Parameters:  map[string]any{"type": "object"},
	}
}

func (e *Executor) connect(ctx context.Context) error {
	c, err := mcpclient.NewStreamableHttpClient(e.url)
	if err != nil {
		return fmt.Errorf("create MCP client: %w", err)
	}
	if err := c.Start(ctx); err != nil {
		_ = c.Close()
		return fmt.Errorf("start MCP client: %w", err)
	}
	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{Name: "voice-agent", Version: "0.1.0"}
	if _, err := c.Initialize(ctx, initReq); err != nil {
		_ = c.Close()
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
	if name == reconnectToolName {
		return e.handleReconnectTool(ctx)
	}
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
	if err != nil && ctx.Err() == nil && e.mode == ReconnectAuto {
		// Transport failure (not a tool-level IsError, not the caller's timeout):
		// the gateway or a backend MCP may have restarted. In auto mode, try a
		// rate-limited reconnect and a single retry. In ask/off mode we surface
		// the error instead (ask mode then offers reconnect_tools to the model).
		if e.reconnect(ctx, false) {
			res, err = e.client.CallTool(ctx, req)
		}
	}
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

// handleReconnectTool services a model-issued reconnect_tools call (ask mode):
// the user has agreed, so it forces a reconnect and reports the outcome as the
// tool result for the model to relay.
func (e *Executor) handleReconnectTool(ctx context.Context) (string, []lemonade.Image, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.reconnect(ctx, true) {
		return "Reconnected to the tools.", nil, nil
	}
	return "", nil, fmt.Errorf("could not reconnect to the tools")
}

// reconnect re-establishes a dead MCP session — the recovery path when a tool
// call fails on the transport (the gateway or a backend MCP restarted). Unless
// forced (a user-authorized reconnect_tools call), it dials at most once per
// backoff window (1s doubling to a 30s cap, reset on success), so a permanently-
// down MCP is retried periodically, never in a tight loop. The caller must hold
// e.mu. Returns true when the client is usable afterwards.
func (e *Executor) reconnect(ctx context.Context, force bool) bool {
	if !force && !e.lastDial.IsZero() && time.Since(e.lastDial) < e.backoff() {
		return false // still cooling down — fail fast instead of hammering
	}
	e.lastDial = time.Now()

	old := e.client
	if err := e.connect(ctx); err != nil { // connect swaps e.client only on success
		e.dialFails++
		log.Printf("mcp: reconnect to %s failed (attempt %d, next try in ~%s): %v", e.url, e.dialFails, e.backoff(), err)
		return false
	}
	e.dialFails = 0
	if old != nil {
		_ = old.Close() // drop the dead session now that a fresh one is up
	}
	log.Printf("mcp: reconnected to %s", e.url)
	return true
}

// backoff is the minimum gap between reconnect attempts: 1s doubling to a 30s cap.
func (e *Executor) backoff() time.Duration {
	const base, maxGap = time.Second, 30 * time.Second
	d := base
	for i := 0; i < e.dialFails && d < maxGap; i++ {
		d *= 2
	}
	if d > maxGap {
		d = maxGap
	}
	return d
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

// ReconnectHint tells the model how to recover a dropped session in ask mode.
// It returns "" in auto/off mode (the model has nothing to do).
func (e *Executor) ReconnectHint() string {
	if e.mode != ReconnectAsk {
		return ""
	}
	return "If a tool call fails because the tools are disconnected, do not retry silently. Tell the user the tools dropped and ask whether to reconnect; only if they agree, call reconnect_tools and then try the request again."
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
