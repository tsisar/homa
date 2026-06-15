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
