// Package lemonade is a thin client for the Lemonade Server's
// OpenAI-compatible API (chat completions with tool-calling + Whisper STT).
package lemonade

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
)

// Client talks to a Lemonade Server. BaseURL is e.g.
// "http://192.168.88.83:8000/api/v1". No auth on the target instance.
type Client struct {
	BaseURL string
	HTTP    *http.Client

	// ChatTemplateKwargs is forwarded as chat_template_kwargs on every chat
	// request. For Qwen3-family reasoning models set {"enable_thinking": false}
	// — otherwise they spend the whole token budget in reasoning_content and
	// return empty content.
	ChatTemplateKwargs map[string]any
}

func New(baseURL string) *Client {
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		HTTP:    &http.Client{}, // no client-wide timeout: use per-call context
	}
}

// Message is a single chat turn. For an assistant turn that calls tools,
// ToolCalls is set; for a tool result, Role=="tool" and ToolCallID is set.
type Message struct {
	Role       string
	Content    string
	ToolCalls  []ToolCall
	ToolCallID string
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
	Content    string         `json:"content,omitempty"`
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
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
	Error *apiError `json:"error"`
}

// Chat sends a non-streaming chat completion. tools may be nil.
func (c *Client) Chat(ctx context.Context, model string, msgs []Message, tools []Tool) (*ChatResult, error) {
	req := chatRequest{
		Model:              model,
		Messages:           toWire(msgs),
		Temperature:        0.7,
		MaxTokens:          512,
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

	resp, err := c.HTTP.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("chat request: %w", err)
	}
	defer resp.Body.Close()

	data, _ := io.ReadAll(resp.Body)
	var cr chatResponse
	if err := json.Unmarshal(data, &cr); err != nil {
		return nil, fmt.Errorf("chat decode (http %d): %w; body: %s", resp.StatusCode, err, snippet(data))
	}
	if cr.Error != nil {
		return nil, fmt.Errorf("lemonade error: %s: %s", cr.Error.Code, cr.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("lemonade http %d: %s", resp.StatusCode, snippet(data))
	}
	if len(cr.Choices) == 0 {
		return nil, fmt.Errorf("lemonade: no choices in response")
	}

	choice := cr.Choices[0]
	res := &ChatResult{
		Content:      strings.TrimSpace(choice.Message.Content),
		FinishReason: choice.FinishReason,
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
		wm := wireMessage{Role: m.Role, Content: m.Content, ToolCallID: m.ToolCallID}
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

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return "", fmt.Errorf("transcribe request: %w", err)
	}
	defer resp.Body.Close()

	data, _ := io.ReadAll(resp.Body)
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

func snippet(b []byte) string {
	const max = 300
	if len(b) > max {
		return string(b[:max]) + "…"
	}
	return string(b)
}
