// Command agent is a thin orchestrator for an English voice assistant:
// text -> Lemonade chat (LLM, optional MCP tool-calling) -> Kokoro TTS -> audio.
//
// Tools come from an MCP endpoint (e.g. a kgateway/agentgateway MCP gateway)
// over Streamable HTTP. Enable with MCP_URL; expose tools with MCP_ALLOW.
// Speech-to-text (Whisper) is wired for the future ESP32 mic client.
//
// Modes:
//
//	go run .                 # HTTP server on :8080 (default)
//	go run . -once "ask..."  # one-shot: ask, print reply, play it locally
//	go run . -say  "text..." # speak the given text locally (no LLM)
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"agent/internal/lemonade"
	"agent/internal/mcp"
	"agent/internal/tts"
)

type config struct {
	lemonadeURL     string // direct Lemonade, used for STT + TTS (the LLM gateway is chat-only)
	chatURL         string // chat-completions endpoint — the kgateway LLM gateway, local provider
	chatModel       string
	sttModel        string
	ttsURL          string
	ttsModel        string
	ttsVoice        string
	ttsFormat       string
	disableThinking bool
	mcpURL          string
	mcpAllow        []string
	searchFiller    string
	addr            string
	systemPrompt    string
}

func loadConfig() config {
	lem := envOr("LEMONADE_URL", "http://192.168.88.83:8000/api/v1")
	return config{
		lemonadeURL:     lem,
		chatURL:         envOr("CHAT_URL", "http://llm.tsisar.local/local/v1"), // gateway, local provider
		chatModel:       envOr("CHAT_MODEL", "Qwen3.6-35B-A3B-MTP-GGUF"),
		sttModel:        envOr("STT_MODEL", "Whisper-Large-v3-Turbo"),
		ttsURL:          envOr("TTS_URL", lem),
		ttsModel:        envOr("TTS_MODEL", "kokoro-v1"),
		ttsVoice:        envOr("TTS_VOICE", "af_heart"),
		ttsFormat:       envOr("TTS_FORMAT", "wav"),
		disableThinking: envOr("DISABLE_THINKING", "true") == "true",
		mcpURL:          envOr("MCP_URL", ""), // empty = MCP disabled
		mcpAllow:        splitCSV(envOr("MCP_ALLOW", "")),
		searchFiller:    envOr("SEARCH_FILLER", "Let me look that up."), // spoken before tools run; empty = off
		addr:            envOr("ADDR", ":8080"),
		systemPrompt:    envOr("SYSTEM_PROMPT", defaultSystemPrompt),
	}
}

const defaultSystemPrompt = "You are Homa, a friendly voice assistant for the home and homelab. " +
	"Always reply in English. " +
	"Keep replies short and conversational, like real speech — usually one to three sentences. " +
	"When the user asks about recent events or facts you are unsure of, use the available tools. " +
	"Do not use markdown, lists, emojis, or special characters, and never read out raw URLs, because your reply will be read aloud."

// version is stamped at build time via -ldflags "-X main.version=...".
var version = "dev"

// toolExecutor is satisfied by *mcp.Executor.
type toolExecutor interface {
	Tools() []lemonade.Tool
	CallTool(ctx context.Context, name, argsJSON string) (string, error)
}

func main() {
	log.SetFlags(log.Ltime)
	once := flag.String("once", "", "one-shot: send this text to the LLM, print and play the reply")
	say := flag.String("say", "", "speak this text locally via TTS (no LLM)")
	flag.Parse()

	cfg := loadConfig()
	ag := &agent{
		cfg: cfg,
		llm: lemonade.New(cfg.chatURL),     // chat via the kgateway LLM gateway (local provider)
		stt: lemonade.New(cfg.lemonadeURL), // STT direct to Lemonade (gateway is chat-only)
		tts: tts.NewOpenAISpeech(cfg.ttsURL, cfg.ttsModel, cfg.ttsVoice, cfg.ttsFormat),
	}
	if cfg.disableThinking {
		// Qwen3-family models reason into reasoning_content and leave content
		// empty unless thinking is disabled — fatal for a voice loop.
		ag.llm.ChatTemplateKwargs = map[string]any{"enable_thinking": false}
	}
	if cfg.mcpURL != "" {
		ex, err := mcp.New(context.Background(), cfg.mcpURL, cfg.mcpAllow)
		if err != nil {
			log.Printf("MCP disabled (%s): %v", cfg.mcpURL, err)
		} else {
			ag.tools = ex
			log.Printf("MCP connected (%s): %d tool(s) exposed: %v", cfg.mcpURL, len(ex.Tools()), toolNames(ex.Tools()))
			if len(ex.Tools()) == 0 {
				log.Printf("note: MCP_ALLOW matched no tools — set it, e.g. MCP_ALLOW=\"ddg_*\"")
			}
		}
	}
	ag.reset()

	switch {
	case *say != "":
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		audio, mime, err := ag.tts.Speak(ctx, *say)
		if err != nil {
			log.Fatalf("tts: %v", err)
		}
		playAudio(audio, mime)
	case *once != "":
		ctx, cancel := context.WithTimeout(context.Background(), 240*time.Second)
		defer cancel()
		reply, audio, mime, err := ag.respond(ctx, *once, func(text string) {
			fmt.Printf("… %s\n", text)
			if fa, fm, ferr := ag.tts.Speak(ctx, text); ferr == nil {
				playAudio(fa, fm)
			}
		})
		if err != nil {
			log.Fatalf("respond: %v", err)
		}
		fmt.Printf("\n🤖 %s\n", reply)
		playAudio(audio, mime)
	default:
		ag.serve()
	}
}

// ---- agent core ----------------------------------------------------------

type agent struct {
	cfg   config
	llm   *lemonade.Client // chat (via the LLM gateway)
	stt   *lemonade.Client // transcription (direct to Lemonade)
	tts   *tts.OpenAISpeech
	tools toolExecutor // nil when MCP is disabled

	mu      sync.Mutex
	history []lemonade.Message
}

func (a *agent) reset() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.history = []lemonade.Message{{Role: "system", Content: a.cfg.systemPrompt}}
}

const maxToolRounds = 5

// respond runs one conversational turn with a tool-calling loop, then
// synthesizes the final reply. onFiller, if set, is called once with a short
// phrase the moment the agent decides to use tools — so the caller can speak a
// "let me check that" cue while the (slow) tool calls run. Returns reply text,
// audio, and its mime.
func (a *agent) respond(ctx context.Context, userText string, onFiller func(string)) (string, []byte, string, error) {
	a.mu.Lock()
	a.history = append(a.history, lemonade.Message{Role: "user", Content: userText})
	a.mu.Unlock()

	var tools []lemonade.Tool
	if a.tools != nil {
		tools = a.tools.Tools()
	}

	reply := ""
	announced := false
	for round := 0; ; round++ {
		a.mu.Lock()
		msgs := append([]lemonade.Message(nil), a.history...)
		a.mu.Unlock()

		// After the budget, stop offering tools so the model must answer.
		offer := tools
		if round >= maxToolRounds {
			offer = nil
		}

		res, err := a.llm.Chat(ctx, a.cfg.chatModel, msgs, offer)
		if err != nil {
			return "", nil, "", fmt.Errorf("llm: %w", err)
		}

		if len(res.ToolCalls) > 0 && offer != nil && a.tools != nil {
			if !announced && onFiller != nil && a.cfg.searchFiller != "" {
				onFiller(a.cfg.searchFiller)
				announced = true
			}
			a.mu.Lock()
			a.history = append(a.history, lemonade.Message{Role: "assistant", Content: res.Content, ToolCalls: res.ToolCalls})
			a.mu.Unlock()
			for _, tc := range res.ToolCalls {
				out, callErr := a.tools.CallTool(ctx, tc.Name, tc.Arguments)
				if callErr != nil {
					out = "error: " + callErr.Error()
				}
				if strings.TrimSpace(out) == "" {
					out = "(no output)"
				}
				log.Printf("tool %s(%s) -> %d chars", tc.Name, truncate(tc.Arguments, 100), len(out))
				a.mu.Lock()
				a.history = append(a.history, lemonade.Message{Role: "tool", ToolCallID: tc.ID, Content: out})
				a.mu.Unlock()
			}
			continue
		}

		reply = res.Content
		break
	}

	if reply == "" {
		reply = "Sorry, I didn't catch that."
	}
	reply = sanitize(reply)

	a.mu.Lock()
	a.history = append(a.history, lemonade.Message{Role: "assistant", Content: reply})
	a.mu.Unlock()

	audio, mime, err := a.tts.Speak(ctx, reply)
	if err != nil {
		return reply, nil, "", fmt.Errorf("tts: %w", err)
	}
	return reply, audio, mime, nil
}

// ---- HTTP server (also the future ESP32 entry point) ---------------------

func (a *agent) serve() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "ok")
	})

	// POST /api/talk {"text": "..."} -> audio reply; X-Reply-Text header (url-encoded).
	mux.HandleFunc("POST /api/talk", func(w http.ResponseWriter, r *http.Request) {
		text, ok := decodeText(w, r)
		if !ok {
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 240*time.Second)
		defer cancel()
		var filler string
		reply, audio, mime, err := a.respond(ctx, text, func(t string) { filler = t })
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		log.Printf("talk: %q -> %q", text, reply)
		if filler != "" {
			w.Header().Set("X-Filler-Text", url.QueryEscape(filler))
		}
		w.Header().Set("X-Reply-Text", url.QueryEscape(reply))
		writeAudio(w, audio, mime)
	})

	// POST /api/say {"text": "..."} -> audio (TTS only, no LLM).
	mux.HandleFunc("POST /api/say", func(w http.ResponseWriter, r *http.Request) {
		text, ok := decodeText(w, r)
		if !ok {
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
		defer cancel()
		audio, mime, err := a.tts.Speak(ctx, text)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		writeAudio(w, audio, mime)
	})

	// POST /api/stt (multipart, field "file" = 16-bit PCM WAV) -> {"text": "..."}.
	mux.HandleFunc("POST /api/stt", func(w http.ResponseWriter, r *http.Request) {
		f, _, err := r.FormFile("file")
		if err != nil {
			http.Error(w, "expected multipart field 'file' (wav)", http.StatusBadRequest)
			return
		}
		defer f.Close()
		wav, _ := io.ReadAll(f)
		ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
		defer cancel()
		text, err := a.stt.Transcribe(ctx, a.cfg.sttModel, wav, "en")
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"text": text})
	})

	// POST /api/reset -> clears conversation history.
	mux.HandleFunc("POST /api/reset", func(w http.ResponseWriter, r *http.Request) {
		a.reset()
		fmt.Fprintln(w, "ok")
	})

	log.Printf("Homa %s listening on %s (LLM=%s, TTS=%s voice=%s)", version, a.cfg.addr, a.cfg.chatModel, a.cfg.ttsModel, a.cfg.ttsVoice)
	if err := http.ListenAndServe(a.cfg.addr, mux); err != nil {
		log.Fatal(err)
	}
}

func decodeText(w http.ResponseWriter, r *http.Request) (string, bool) {
	var req struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || strings.TrimSpace(req.Text) == "" {
		http.Error(w, "expected JSON {\"text\": \"...\"}", http.StatusBadRequest)
		return "", false
	}
	return req.Text, true
}

func writeAudio(w http.ResponseWriter, audio []byte, mime string) {
	w.Header().Set("Content-Type", mime)
	_, _ = w.Write(audio)
}

// ---- local playback (macOS afplay) ---------------------------------------

func playAudio(audio []byte, mime string) {
	ext := ".wav"
	if strings.Contains(mime, "mpeg") || strings.Contains(mime, "mp3") {
		ext = ".mp3"
	}
	f, err := os.CreateTemp("", "agent-*"+ext)
	if err != nil {
		log.Printf("tempfile: %v", err)
		return
	}
	defer os.Remove(f.Name())
	if _, err := f.Write(audio); err != nil {
		log.Printf("write audio: %v", err)
		return
	}
	f.Close()
	if err := exec.Command("afplay", f.Name()).Run(); err != nil {
		log.Printf("afplay: %v (audio saved at %s)", err, f.Name())
	}
}

// ---- misc ----------------------------------------------------------------

// sanitize strips markdown-ish characters the model may emit despite the
// system prompt, so they aren't read aloud.
func sanitize(s string) string {
	r := strings.NewReplacer("*", "", "#", "", "`", "", "_", "")
	return strings.TrimSpace(r.Replace(s))
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func splitCSV(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func toolNames(ts []lemonade.Tool) []string {
	n := make([]string, len(ts))
	for i, t := range ts {
		n[i] = t.Name
	}
	return n
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}
