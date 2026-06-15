package lemonade

import "testing"

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
