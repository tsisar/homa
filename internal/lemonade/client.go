// Package lemonade is a thin client for the Lemonade Server's
// OpenAI-compatible API (chat completions with tool-calling + Whisper STT).
package lemonade

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"regexp"
	"strings"

	"github.com/tsisar/extended-log-go/log"
)

// ErrContextLength is wrapped into the error returned by Chat when the backend
// rejects a request because the prompt exceeds the model's context window.
// Callers can detect it with errors.Is and compact the conversation.
var ErrContextLength = errors.New("context length exceeded")

// Client talks to a Lemonade Server (directly or via the LLM gateway, which
// proxies the same OpenAI-compatible API). BaseURL is e.g.
// "http://llm.tsisar.local/local/v1". No auth on the target instance.
type Client struct {
	BaseURL string
	HTTP    *http.Client

	// ChatTemplateKwargs is forwarded as chat_template_kwargs on every chat
	// request. For Qwen3-family reasoning models set {"enable_thinking": false}
	// — otherwise they spend the whole token budget in reasoning_content and
	// return empty content.
	ChatTemplateKwargs map[string]any

	// MaxTokens caps the reply length (max_tokens). Zero falls back to 512.
	MaxTokens int
}

func New(baseURL string) *Client {
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		HTTP:    &http.Client{}, // no client-wide timeout: use per-call context
	}
}

// Message is a single chat turn. For an assistant turn that calls tools,
// ToolCalls is set; for a tool result, Role=="tool" and ToolCallID is set.
// Images, when present, are sent alongside Content as multimodal parts — used
// for tool results that return a picture (e.g. a rendered Grafana panel) so a
// vision model can read them.
type Message struct {
	Role       string
	Content    string
	ToolCalls  []ToolCall
	ToolCallID string
	Images     []Image
}

// Image is an inline image attached to a message. Data is base64-encoded (no
// data: prefix); MIME is like "image/png".
type Image struct {
	MIME string
	Data string
}

// Tool is a function the model may call (OpenAI tool schema).
type Tool struct {
	Name        string
	Description string
	Parameters  map[string]any // JSON Schema
}

// ToolCall is a model's request to invoke a tool.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string // raw JSON
}

// ChatResult is the outcome of one chat completion.
type ChatResult struct {
	Content      string
	ToolCalls    []ToolCall
	FinishReason string

	// PromptTokens and CompletionTokens are the server-reported usage for this
	// call. PromptTokens is the exact size of what was sent — the authoritative
	// signal for deciding when the conversation must be compacted.
	PromptTokens     int
	CompletionTokens int
}

type apiError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Type    string `json:"type"`
}

// ---- wire types (OpenAI chat-completions shape) --------------------------

type wireFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type wireToolCall struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Function wireFunc `json:"function"`
}

type wireMessage struct {
	Role       string         `json:"role"`
	Content    any            `json:"content,omitempty"` // string, or []wireContentPart when images are attached
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type wireImageURL struct {
	URL string `json:"url"`
}

type wireContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *wireImageURL `json:"image_url,omitempty"`
}

type wireToolFunc struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type wireTool struct {
	Type     string       `json:"type"`
	Function wireToolFunc `json:"function"`
}

type chatRequest struct {
	Model              string         `json:"model"`
	Messages           []wireMessage  `json:"messages"`
	Tools              []wireTool     `json:"tools,omitempty"`
	Temperature        float64        `json:"temperature,omitempty"`
	MaxTokens          int            `json:"max_tokens,omitempty"`
	Stream             bool           `json:"stream"`
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
			ToolCalls        []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	Error *apiError `json:"error"`
}

// Chat sends a non-streaming chat completion. tools may be nil.
func (c *Client) Chat(ctx context.Context, model string, msgs []Message, tools []Tool) (*ChatResult, error) {
	maxTokens := c.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}
	req := chatRequest{
		Model:              model,
		Messages:           toWire(msgs),
		Temperature:        0.7,
		MaxTokens:          maxTokens,
		Stream:             false,
		ChatTemplateKwargs: c.ChatTemplateKwargs,
	}
	for _, t := range tools {
		req.Tools = append(req.Tools, wireTool{
			Type:     "function",
			Function: wireToolFunc{Name: t.Name, Description: t.Description, Parameters: t.Parameters},
		})
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// Trace: full request to the model (mirrors alert-agent's llm clients), with
	// inline image payloads elided so a rendered panel doesn't flood the log.
	log.Tracef("[llm] request: %s", redactImages(body))

	resp, err := c.HTTP.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("chat request: %w", err)
	}
	defer resp.Body.Close()

	data, _ := io.ReadAll(resp.Body)
	// Trace: full response from the model.
	log.Tracef("[llm] response: %s", data)
	var cr chatResponse
	if err := json.Unmarshal(data, &cr); err != nil {
		return nil, fmt.Errorf("chat decode (http %d): %w; body: %s", resp.StatusCode, err, snippet(data))
	}
	if cr.Error != nil {
		if cr.Error.Code == "context_length_exceeded" || mentionsContextLimit(cr.Error.Message) {
			return nil, fmt.Errorf("lemonade error: %s: %s: %w", cr.Error.Code, cr.Error.Message, ErrContextLength)
		}
		return nil, fmt.Errorf("lemonade error: %s: %s", cr.Error.Code, cr.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		// Some gateways reshape the overflow into a bare non-200 whose body is not
		// the OpenAI error shape; still surface it as ErrContextLength so the
		// caller can compact and retry instead of wedging.
		if (resp.StatusCode == http.StatusBadRequest || resp.StatusCode == http.StatusRequestEntityTooLarge) && mentionsContextLimit(string(data)) {
			return nil, fmt.Errorf("lemonade http %d: %s: %w", resp.StatusCode, snippet(data), ErrContextLength)
		}
		return nil, fmt.Errorf("lemonade http %d: %s", resp.StatusCode, snippet(data))
	}
	if len(cr.Choices) == 0 {
		return nil, fmt.Errorf("lemonade: no choices in response")
	}

	choice := cr.Choices[0]
	res := &ChatResult{
		Content:          strings.TrimSpace(choice.Message.Content),
		FinishReason:     choice.FinishReason,
		PromptTokens:     cr.Usage.PromptTokens,
		CompletionTokens: cr.Usage.CompletionTokens,
	}
	for _, tc := range choice.Message.ToolCalls {
		res.ToolCalls = append(res.ToolCalls, ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
	}

	// A normal (non-tool) turn with empty content usually means the model
	// reasoned but never answered — surface a clear hint.
	if len(res.ToolCalls) == 0 && res.Content == "" {
		if choice.Message.ReasoningContent != "" {
			return nil, fmt.Errorf("lemonade: empty content but model produced reasoning — disable thinking (chat_template_kwargs.enable_thinking=false) or raise max_tokens")
		}
		return nil, fmt.Errorf("lemonade: model returned empty content")
	}
	return res, nil
}

func toWire(msgs []Message) []wireMessage {
	out := make([]wireMessage, len(msgs))
	for i, m := range msgs {
		wm := wireMessage{Role: m.Role, ToolCallID: m.ToolCallID}
		switch {
		case len(m.Images) > 0:
			// Multimodal: emit a content array of text + image parts.
			parts := make([]wireContentPart, 0, len(m.Images)+1)
			if m.Content != "" {
				parts = append(parts, wireContentPart{Type: "text", Text: m.Content})
			}
			for _, img := range m.Images {
				parts = append(parts, wireContentPart{
					Type:     "image_url",
					ImageURL: &wireImageURL{URL: "data:" + img.MIME + ";base64," + img.Data},
				})
			}
			wm.Content = parts
		case m.Content != "":
			wm.Content = m.Content
		}
		for _, tc := range m.ToolCalls {
			wm.ToolCalls = append(wm.ToolCalls, wireToolCall{
				ID:       tc.ID,
				Type:     "function",
				Function: wireFunc{Name: tc.Name, Arguments: tc.Arguments},
			})
		}
		out[i] = wm
	}
	return out
}

// Transcribe sends a 16-bit PCM WAV to Whisper and returns the text.
// lang is an optional ISO-639-1 hint (e.g. "en"); empty means auto-detect.
func (c *Client) Transcribe(ctx context.Context, model string, wav []byte, lang string) (string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	fw, err := w.CreateFormFile("file", "audio.wav")
	if err != nil {
		return "", err
	}
	if _, err := fw.Write(wav); err != nil {
		return "", err
	}
	_ = w.WriteField("model", model)
	if lang != "" {
		_ = w.WriteField("language", lang)
	}
	if err := w.Close(); err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/audio/transcriptions", &buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())

	// Trace: STT request metadata only — the body is a raw WAV blob.
	log.Tracef("[stt] request: model=%s lang=%s wav=%dB", model, lang, len(wav))

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return "", fmt.Errorf("transcribe request: %w", err)
	}
	defer resp.Body.Close()

	data, _ := io.ReadAll(resp.Body)
	// Trace: STT response is small JSON ({"text":...}) — safe to log in full.
	log.Tracef("[stt] response: %s", data)
	var tr struct {
		Text  string    `json:"text"`
		Error *apiError `json:"error"`
	}
	if err := json.Unmarshal(data, &tr); err != nil {
		return "", fmt.Errorf("transcribe decode (http %d): %w; body: %s", resp.StatusCode, err, snippet(data))
	}
	if tr.Error != nil {
		return "", fmt.Errorf("lemonade error: %s: %s", tr.Error.Code, tr.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("lemonade http %d: %s", resp.StatusCode, snippet(data))
	}
	return strings.TrimSpace(tr.Text), nil
}

// dataURIRe matches a base64 image data URI value inside JSON so its payload can
// be elided from trace logs (keeping the "data:<mime>;base64," prefix).
var dataURIRe = regexp.MustCompile(`("url":"data:[^"]*?base64,)[A-Za-z0-9+/=]+"`)

// redactImages strips base64 image payloads from a marshaled request for logging.
func redactImages(b []byte) string {
	if !bytes.Contains(b, []byte("base64,")) {
		return string(b)
	}
	return string(dataURIRe.ReplaceAll(b, []byte(`$1…"`)))
}

// mentionsContextLimit reports whether s reads like a context-window overflow.
// Backends and gateways word this differently, so match on several phrasings
// rather than a single literal.
func mentionsContextLimit(s string) bool {
	s = strings.ToLower(s)
	for _, k := range []string{
		"exceeds the available context",
		"context length",
		"maximum context",
		"context window",
		"n_ctx",
		"too many tokens",
	} {
		if strings.Contains(s, k) {
			return true
		}
	}
	return false
}

func snippet(b []byte) string {
	const max = 300
	if len(b) > max {
		return string(b[:max]) + "…"
	}
	return string(b)
}
