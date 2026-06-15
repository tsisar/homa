package lemonade

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestToWireMultimodal(t *testing.T) {
	msgs := []Message{
		{Role: "tool", ToolCallID: "c1", Content: "panel:", Images: []Image{{MIME: "image/png", Data: "QUJD"}}},
		{Role: "user", Content: "hi"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c1", Name: "t", Arguments: "{}"}}},
	}
	w := toWire(msgs)

	parts, ok := w[0].Content.([]wireContentPart)
	if !ok {
		t.Fatalf("image message content type = %T, want []wireContentPart", w[0].Content)
	}
	if len(parts) != 2 || parts[0].Type != "text" || parts[1].Type != "image_url" {
		t.Fatalf("parts = %+v", parts)
	}
	if want := "data:image/png;base64,QUJD"; parts[1].ImageURL == nil || parts[1].ImageURL.URL != want {
		t.Errorf("image url = %v, want %q", parts[1].ImageURL, want)
	}

	if s, ok := w[1].Content.(string); !ok || s != "hi" {
		t.Errorf("plain content = %v (%T), want \"hi\"", w[1].Content, w[1].Content)
	}

	// assistant tool-call message: content omitted, tool call preserved
	if w[2].Content != nil {
		t.Errorf("assistant content = %v, want nil", w[2].Content)
	}
	if len(w[2].ToolCalls) != 1 || w[2].ToolCalls[0].Function.Name != "t" {
		t.Errorf("tool calls = %+v", w[2].ToolCalls)
	}

	// the image message must serialize as a content array carrying the data URI
	b, err := json.Marshal(w[0])
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"type":"text"`, `"type":"image_url"`, `"url":"data:image/png;base64,QUJD"`} {
		if !strings.Contains(string(b), want) {
			t.Errorf("marshaled message missing %q: %s", want, b)
		}
	}
}

func TestRedactImages(t *testing.T) {
	plain := []byte(`{"messages":[{"role":"user","content":"hello"}]}`)
	if got := redactImages(plain); got != string(plain) {
		t.Errorf("redactImages changed an image-free body: %s", got)
	}

	payload := "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo="
	withImg := []byte(`{"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,` + payload + `"}}]}`)
	got := redactImages(withImg)
	if !strings.Contains(got, "data:image/png;base64,") {
		t.Errorf("redacted body lost the mime prefix: %s", got)
	}
	if strings.Contains(got, payload) {
		t.Errorf("redacted body still contains the base64 payload: %s", got)
	}
}

func TestMentionsContextLimit(t *testing.T) {
	overflow := []string{
		"request (4112 tokens) exceeds the available context size (4096 tokens)",
		"This model's maximum context length is 4096 tokens",
		"Context window exceeded",
		"n_ctx too small for this prompt",
		"too many tokens in the request",
	}
	other := []string{
		"rate limit exceeded",
		"invalid api key",
		"connection refused",
		"",
	}
	for _, s := range overflow {
		if !mentionsContextLimit(s) {
			t.Errorf("mentionsContextLimit(%q) = false, want true", s)
		}
	}
	for _, s := range other {
		if mentionsContextLimit(s) {
			t.Errorf("mentionsContextLimit(%q) = true, want false", s)
		}
	}
}
