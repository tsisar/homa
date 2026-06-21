package main

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"agent/internal/lemonade"
)

// agentWith returns an agent whose image cache holds n images img1..imgN with
// data DATA1..DATAN, registered through the real rememberImages path.
func agentWith(t *testing.T, n int) *agent {
	t.Helper()
	a := &agent{}
	for i := 0; i < n; i++ {
		a.rememberImages([]lemonade.Image{{MIME: "image/png", Data: fmt.Sprintf("DATA%d", i+1)}})
	}
	return a
}

func decodeObj(t *testing.T, js string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(js), &m); err != nil {
		t.Fatalf("result is not a JSON object: %v (%s)", err, js)
	}
	return m
}

// TestResolveImageArgs covers the send-tool bridge: a handle resolves to the real
// bytes, a missing source auto-injects the latest image, and a model-chosen real
// source / non-send tool / empty cache all pass through untouched.
func TestResolveImageArgs(t *testing.T) {
	const photo = "telegram_telegram_send_photo"

	t.Run("handle in base64 resolves to bytes, caption kept", func(t *testing.T) {
		a := agentWith(t, 2)
		m := decodeObj(t, a.resolveImageArgs(photo, `{"base64":"img1","caption":"hi"}`))
		if m["base64"] != "DATA1" {
			t.Errorf("base64 = %v, want DATA1", m["base64"])
		}
		if m["caption"] != "hi" {
			t.Errorf("caption = %v, want hi", m["caption"])
		}
	})

	t.Run("handle in url moves to base64, url dropped", func(t *testing.T) {
		a := agentWith(t, 2)
		m := decodeObj(t, a.resolveImageArgs(photo, `{"url":"img2"}`))
		if m["base64"] != "DATA2" {
			t.Errorf("base64 = %v, want DATA2", m["base64"])
		}
		if _, ok := m["url"]; ok {
			t.Errorf("url should be removed, got %v", m["url"])
		}
	})

	t.Run("missing source auto-injects latest", func(t *testing.T) {
		a := agentWith(t, 2)
		m := decodeObj(t, a.resolveImageArgs(photo, `{"caption":"hi"}`))
		if m["base64"] != "DATA2" {
			t.Errorf("base64 = %v, want DATA2 (latest)", m["base64"])
		}
	})

	t.Run("empty args auto-inject latest", func(t *testing.T) {
		a := agentWith(t, 1)
		m := decodeObj(t, a.resolveImageArgs(photo, ""))
		if m["base64"] != "DATA1" {
			t.Errorf("base64 = %v, want DATA1", m["base64"])
		}
	})

	t.Run("unresolved handle injects latest and clears junk", func(t *testing.T) {
		a := agentWith(t, 1)
		m := decodeObj(t, a.resolveImageArgs(photo, `{"base64":"img9"}`))
		if m["base64"] != "DATA1" {
			t.Errorf("base64 = %v, want DATA1", m["base64"])
		}
	})

	t.Run("unresolved handle beside a real path keeps a single source", func(t *testing.T) {
		a := agentWith(t, 1) // only img1 exists
		m := decodeObj(t, a.resolveImageArgs(photo, `{"base64":"img9","path":"/tmp/real.png"}`))
		if _, ok := m["base64"]; ok {
			t.Errorf("junk handle base64 should be dropped, got %v", m["base64"])
		}
		if m["path"] != "/tmp/real.png" {
			t.Errorf("real path = %v, want /tmp/real.png", m["path"])
		}
	})

	t.Run("real base64 beside a handle url keeps a single source", func(t *testing.T) {
		a := agentWith(t, 1)
		m := decodeObj(t, a.resolveImageArgs(photo, `{"base64":"AAAArealbytes","url":"img9"}`))
		if _, ok := m["url"]; ok {
			t.Errorf("handle url should be dropped, got %v", m["url"])
		}
		if m["base64"] != "AAAArealbytes" {
			t.Errorf("real base64 = %v, want AAAArealbytes", m["base64"])
		}
	})

	t.Run("empty-string source is treated as missing and injected", func(t *testing.T) {
		a := agentWith(t, 1)
		m := decodeObj(t, a.resolveImageArgs(photo, `{"base64":"","caption":"x"}`))
		if m["base64"] != "DATA1" {
			t.Errorf("base64 = %v, want DATA1", m["base64"])
		}
	})

	t.Run("album does not auto-inject into a source-less item", func(t *testing.T) {
		a := agentWith(t, 2)
		out := a.resolveImageArgs("telegram_telegram_send_album",
			`{"items":[{"base64":"img1"},{"caption":"no source"}]}`)
		m := decodeObj(t, out)
		items := m["items"].([]any)
		if it0 := items[0].(map[string]any); it0["base64"] != "DATA1" {
			t.Errorf("items[0].base64 = %v, want DATA1", it0["base64"])
		}
		if it1 := items[1].(map[string]any); it1["base64"] != nil {
			t.Errorf("items[1] should stay source-less, got base64=%v", it1["base64"])
		}
	})

	t.Run("real url is left untouched", func(t *testing.T) {
		a := agentWith(t, 1)
		in := `{"url":"https://example.com/p.png"}`
		out := a.resolveImageArgs(photo, in)
		m := decodeObj(t, out)
		if m["url"] != "https://example.com/p.png" {
			t.Errorf("url = %v, want the original URL", m["url"])
		}
		if _, ok := m["base64"]; ok {
			t.Errorf("base64 should not be injected over a real url, got %v", m["base64"])
		}
	})

	t.Run("non-send tool passes through verbatim", func(t *testing.T) {
		a := agentWith(t, 1)
		in := `{"base64":"img1"}`
		if out := a.resolveImageArgs("grafana_get_panel_image", in); out != in {
			t.Errorf("non-send tool rewritten: %q", out)
		}
	})

	t.Run("empty cache passes through verbatim", func(t *testing.T) {
		a := &agent{}
		in := `{"caption":"x"}`
		if out := a.resolveImageArgs(photo, in); out != in {
			t.Errorf("rewritten with empty cache: %q", out)
		}
	})

	t.Run("album resolves each item's handle", func(t *testing.T) {
		a := agentWith(t, 2)
		out := a.resolveImageArgs("telegram_telegram_send_album", `{"items":[{"base64":"img1"},{"base64":"img2"}]}`)
		m := decodeObj(t, out)
		items, ok := m["items"].([]any)
		if !ok || len(items) != 2 {
			t.Fatalf("items = %v", m["items"])
		}
		for i, want := range []string{"DATA1", "DATA2"} {
			it := items[i].(map[string]any)
			if it["base64"] != want {
				t.Errorf("items[%d].base64 = %v, want %v", i, it["base64"], want)
			}
		}
	})
}

func TestLooksLikeHandle(t *testing.T) {
	tests := []struct {
		in   string
		want bool
	}{
		{"img1", true},
		{"img42", true},
		{" img3 ", true},
		{"img", false},
		{"image", false},
		{"imgabc", false},
		{"img1x", false},
		{"https://x/p.png", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := looksLikeHandle(tt.in); got != tt.want {
			t.Errorf("looksLikeHandle(%q) = %v, want %v", tt.in, got, tt.want)
		}
	}
}

// TestRememberImages checks handle assignment, the returned hint, and that the
// cache is bounded to the most recent maxCachedImages.
func TestRememberImages(t *testing.T) {
	t.Run("returns a hint naming the handle", func(t *testing.T) {
		a := &agent{}
		hint := a.rememberImages([]lemonade.Image{{MIME: "image/png", Data: "X"}})
		if !strings.Contains(hint, "img1") {
			t.Errorf("hint missing handle: %q", hint)
		}
		if a.imgCache["img1"].Data != "X" {
			t.Errorf("img1 not cached: %+v", a.imgCache)
		}
	})

	t.Run("no images -> no hint", func(t *testing.T) {
		if h := (&agent{}).rememberImages(nil); h != "" {
			t.Errorf("hint = %q, want empty", h)
		}
	})

	t.Run("cache bounded to maxCachedImages, oldest evicted", func(t *testing.T) {
		a := &agent{}
		total := maxCachedImages + 5
		for i := 0; i < total; i++ {
			a.rememberImages([]lemonade.Image{{MIME: "image/png", Data: fmt.Sprintf("D%d", i)}})
		}
		if len(a.imgOrder) != maxCachedImages || len(a.imgCache) != maxCachedImages {
			t.Fatalf("cache size = %d/%d, want %d", len(a.imgOrder), len(a.imgCache), maxCachedImages)
		}
		if _, ok := a.imgCache["img1"]; ok {
			t.Error("img1 should have been evicted")
		}
		last := fmt.Sprintf("img%d", total)
		if _, ok := a.imgCache[last]; !ok {
			t.Errorf("latest %s should be present", last)
		}
	})
}

// TestValidToolCalls drops only the calls whose arguments are malformed JSON
// (the truncated-base64 case), keeping valid and empty-argument calls.
func TestValidToolCalls(t *testing.T) {
	in := []lemonade.ToolCall{
		{ID: "a", Name: "ok", Arguments: `{"x":1}`},
		{ID: "b", Name: "noargs", Arguments: ``},
		{ID: "c", Name: "truncated", Arguments: `{"base64":"iVBORw0KGgoAAA`},
		{ID: "d", Name: "alsoOK", Arguments: `{"base64":"img1"}`},
	}
	got := validToolCalls(in)
	want := []string{"a", "b", "d"}
	if len(got) != len(want) {
		t.Fatalf("kept %d calls, want %d: %+v", len(got), len(want), got)
	}
	for i, id := range want {
		if got[i].ID != id {
			t.Errorf("kept[%d] = %s, want %s", i, got[i].ID, id)
		}
	}
}

// TestResetClearsImageCache guards that a conversation reset drops cached images
// and the handle counter, so handles never leak across conversations.
func TestResetClearsImageCache(t *testing.T) {
	a := agentWith(t, 3)
	a.reset()
	if a.imgCache != nil || a.imgOrder != nil || a.imgSeq != 0 {
		t.Errorf("after reset: cache=%v order=%v seq=%d", a.imgCache, a.imgOrder, a.imgSeq)
	}
	// Next image starts at img1 again.
	a.rememberImages([]lemonade.Image{{MIME: "image/png", Data: "Y"}})
	if _, ok := a.imgCache["img1"]; !ok {
		t.Errorf("handle counter not reset: %v", a.imgCache)
	}
}
