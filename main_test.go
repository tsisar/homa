package main

import (
	"strings"
	"testing"

	"agent/internal/lemonade"
)

func TestClampText(t *testing.T) {
	tests := []struct {
		name  string
		in    string
		limit int
		want  int // expected length of the result
	}{
		{"disabled", "abcdef", 0, 6},
		{"under", "abc", 10, 3},
		{"exact", "abcde", 5, 5},
		{"over", strings.Repeat("x", 1000), 500, 500},
		{"limit below marker", "abcdefghij", 5, 5},
	}
	for _, tt := range tests {
		got := clampText(tt.in, tt.limit)
		if len(got) != tt.want {
			t.Errorf("%s: clampText len = %d, want %d", tt.name, len(got), tt.want)
		}
		// Clamping the result again must be a no-op, or the chat() retry loop
		// could re-clip the same value forever without shrinking it.
		if again := clampText(got, tt.limit); again != got {
			t.Errorf("%s: clampText not idempotent: %q -> %q", tt.name, got, again)
		}
	}
}

func msg(role string, toolCall bool) lemonade.Message {
	m := lemonade.Message{Role: role, Content: role}
	if toolCall {
		m.ToolCalls = []lemonade.ToolCall{{ID: "x", Name: "t"}}
	}
	return m
}

// validKeptWindow checks the kept slice is a valid chat sequence: it starts at a
// user turn and every tool message sits inside an open assistant tool-call group.
func validKeptWindow(keep []lemonade.Message) bool {
	if len(keep) == 0 || keep[0].Role != "user" {
		return false
	}
	open := false
	for _, m := range keep {
		switch m.Role {
		case "user":
			open = false
		case "assistant":
			open = len(m.ToolCalls) > 0
		case "tool":
			if !open {
				return false
			}
		}
	}
	return true
}

func TestSplitForCompaction(t *testing.T) {
	tests := []struct {
		name   string
		h      []lemonade.Message
		wantOK bool
	}{
		{"single user", []lemonade.Message{msg("user", false)}, false},
		{"one full turn", []lemonade.Message{msg("user", false), msg("assistant", false)}, false},
		{
			name: "four plain turns",
			h: []lemonade.Message{
				msg("user", false), msg("assistant", false),
				msg("user", false), msg("assistant", false),
				msg("user", false), msg("assistant", false),
				msg("user", false), msg("assistant", false),
			},
			wantOK: true,
		},
		{
			name: "turn with tool group",
			h: []lemonade.Message{
				msg("user", false), msg("assistant", true), msg("tool", false), msg("assistant", false),
				msg("user", false), msg("assistant", false),
			},
			wantOK: true,
		},
	}
	for _, tt := range tests {
		older, keep, ok := splitForCompaction(tt.h)
		if ok != tt.wantOK {
			t.Errorf("%s: ok = %v, want %v", tt.name, ok, tt.wantOK)
			continue
		}
		if !ok {
			continue
		}
		if len(older)+len(keep) != len(tt.h) {
			t.Errorf("%s: older+keep = %d, want %d", tt.name, len(older)+len(keep), len(tt.h))
		}
		if !validKeptWindow(keep) {
			t.Errorf("%s: kept window is not a valid chat sequence: %+v", tt.name, keep)
		}
	}
}

func TestTruncateOversizedProgressThenStops(t *testing.T) {
	a := &agent{}
	a.history = []lemonade.Message{
		{Role: "user", Content: "hi"},
		{Role: "tool", Content: strings.Repeat("x", 5000)},
		{Role: "tool", Content: strings.Repeat("y", 9000)},
	}
	if !a.truncateOversized() {
		t.Fatal("first truncateOversized should make progress")
	}
	for i, m := range a.history {
		if len(m.Content) > toolDigestLimit {
			t.Errorf("history[%d] is %d bytes, want <= %d", i, len(m.Content), toolDigestLimit)
		}
	}
	if a.truncateOversized() {
		t.Error("second truncateOversized should report no progress")
	}
}

func TestLooksLikeToolCall(t *testing.T) {
	// the exact shape leaked into content in the wild (Qwen XML tool call)
	leaked := "<tool_call>\n<function=grafana_query_prometheus>\n<parameter=datasourceUid>\nPBFA97CFB590B2093\n</parameter>\n</function>\n</tool_call>"
	toolCalls := []string{
		leaked,
		"<function=foo>\n</function>",
		`{"name":"grafana_query_prometheus","arguments":{"expr":"up"}}`,
		`  {"arguments": {}, "name": "x"}  `,
	}
	answers := []string{
		"The median request time is about half a second.",
		"I couldn't find that metric, sorry.",
		"",
		"I used the search function to look it up.", // mentions "function" but is real speech
	}
	for _, s := range toolCalls {
		if !looksLikeToolCall(s) {
			t.Errorf("looksLikeToolCall(%q) = false, want true", truncate(s, 40))
		}
	}
	for _, s := range answers {
		if looksLikeToolCall(s) {
			t.Errorf("looksLikeToolCall(%q) = true, want false", truncate(s, 40))
		}
	}
}

func TestTruncateOversizedClipsSummary(t *testing.T) {
	a := &agent{}
	a.summary = strings.Repeat("z", 4000) // nothing in history; summary alone is too big
	if !a.truncateOversized() {
		t.Fatal("summary clip should make progress")
	}
	if len(a.summary) > summaryCharLimit {
		t.Errorf("summary is %d bytes, want <= %d", len(a.summary), summaryCharLimit)
	}
	if a.truncateOversized() {
		t.Error("second truncateOversized should report no progress")
	}
}
