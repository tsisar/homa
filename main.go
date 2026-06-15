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
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"agent/internal/lemonade"
	"agent/internal/mcp"
	"agent/internal/tts"

	"github.com/tsisar/extended-log-go/log"
)

type config struct {
	apiURL          string // single OpenAI-compatible endpoint for chat + STT + TTS (the LLM gateway, local provider)
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
	contextTokens   int // model context window; history is summarized before it fills
	maxTokens       int // reply budget reserved below the context window
	toolOutputLimit int // chars a single tool result may add to live history
	maxToolRounds   int // tool-calling rounds before forcing a final answer
}

func loadConfig() config {
	// One endpoint for everything. Defaults to the LLM gateway (local provider),
	// which now fronts audio (TTS/STT) as well as chat; point API_URL straight at
	// Lemonade (http://192.168.88.83:8000/api/v1) for local/offline runs.
	api := envOr("API_URL", "http://llm.tsisar.local/local/v1")
	return config{
		apiURL:          api,
		chatModel:       envOr("CHAT_MODEL", "Qwen3.6-35B-A3B-MTP-GGUF"),
		sttModel:        envOr("STT_MODEL", "Whisper-Large-v3-Turbo"),
		ttsURL:          envOr("TTS_URL", api), // optional override (e.g. Chatterbox); defaults to API_URL
		ttsModel:        envOr("TTS_MODEL", "kokoro-v1"),
		ttsVoice:        envOr("TTS_VOICE", "af_heart"),
		ttsFormat:       envOr("TTS_FORMAT", "wav"),
		disableThinking: envOr("DISABLE_THINKING", "true") == "true",
		mcpURL:          envOr("MCP_URL", ""), // empty = MCP disabled
		mcpAllow:        splitCSV(envOr("MCP_ALLOW", "")),
		searchFiller:    envOr("SEARCH_FILLER", "Let me look that up."), // spoken before tools run; empty = off
		addr:            envOr("ADDR", ":8080"),
		systemPrompt:    envOr("SYSTEM_PROMPT", defaultSystemPrompt),
		// CONTEXT_TOKENS must match the backend's served context (llama-server
		// -c / --ctx-size). The default mirrors a conservative 4096 window.
		contextTokens:   envInt("CONTEXT_TOKENS", 4096),
		maxTokens:       envInt("MAX_TOKENS", 512),
		toolOutputLimit: envInt("TOOL_OUTPUT_LIMIT", 8000),
		maxToolRounds:   envInt("MAX_TOOL_ROUNDS", 8),
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
	CallTool(ctx context.Context, name, argsJSON string) (string, []lemonade.Image, error)
}

func main() {
	once := flag.String("once", "", "one-shot: send this text to the LLM, print and play the reply")
	say := flag.String("say", "", "speak this text locally via TTS (no LLM)")
	flag.Parse()

	cfg := loadConfig()
	if cfg.maxTokens+ctxSafetyMargin >= cfg.contextTokens {
		log.Printf("warning: MAX_TOKENS (%d) + margin >= CONTEXT_TOKENS (%d); proactive compaction is disabled, relying on the reactive path", cfg.maxTokens, cfg.contextTokens)
	}
	ag := &agent{
		cfg: cfg,
		lem: lemonade.New(cfg.apiURL), // chat + STT, both via the single endpoint
		tts: tts.NewOpenAISpeech(cfg.ttsURL, cfg.ttsModel, cfg.ttsVoice, cfg.ttsFormat),
	}
	ag.lem.MaxTokens = cfg.maxTokens
	if cfg.disableThinking {
		// Qwen3-family models reason into reasoning_content and leave content
		// empty unless thinking is disabled — fatal for a voice loop.
		ag.lem.ChatTemplateKwargs = map[string]any{"enable_thinking": false}
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
		reply, audio, mime, err := ag.respond(ctx, *once, "", func(text string) {
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
	lem   *lemonade.Client // chat + transcription, both via the single endpoint
	tts   *tts.OpenAISpeech
	tools toolExecutor // nil when MCP is disabled

	// turnMu serializes whole turns. compact() releases a.mu across a slow
	// summarize() call, so without this two concurrent turns (or a reset racing
	// a turn) could interleave and drop live history. a.mu still guards the
	// short critical sections; lock order is always turnMu before a.mu.
	turnMu sync.Mutex

	mu sync.Mutex
	// summary is a rolling memory of turns that were evicted to free context;
	// history holds the recent turns verbatim. The system prompt is not stored
	// here — messages() prepends it (and the summary) on every request.
	summary    string
	history    []lemonade.Message
	lastPrompt int // prompt_tokens of the most recent completion
}

// reset clears the conversation and returns how many history messages were
// dropped, so callers can log what the wipe affected.
func (a *agent) reset() int {
	a.turnMu.Lock() // wait for any in-flight turn so the wipe is authoritative
	defer a.turnMu.Unlock()
	a.mu.Lock()
	defer a.mu.Unlock()
	n := len(a.history)
	a.summary = ""
	a.history = nil
	a.lastPrompt = 0
	return n
}

const (
	keepRecent        = 6    // recent messages kept verbatim when compacting
	ctxSafetyMargin   = 256  // tokens left free below the window
	toolDigestLimit   = 500  // chars of a tool result kept in summaries / last-resort truncation
	summaryCharLimit  = 1500 // hard cap on the rolling summary so it can't grow into the window
	maxCompactRetries = 4    // compaction attempts on a context-overflow before giving up
)

// finalAnswerNudge steers the model to answer in words on the round where tools
// are withdrawn, so it doesn't emit another (text) tool call to be read aloud.
const finalAnswerNudge = "Answer my question now in one or two plain spoken sentences, using what you already have. Do not call any tools or output any tool-call syntax."

// messages assembles the full request: a single system message (the prompt plus
// the rolling summary, if any) followed by the recent turns. The summary is
// folded into the one system message rather than added as a second one, since
// some chat templates only treat the first system message specially. Callers
// must hold a.mu.
func (a *agent) messages() []lemonade.Message {
	system := a.cfg.systemPrompt
	if a.summary != "" {
		system += "\n\nSummary of the conversation so far:\n" + a.summary
	}
	out := make([]lemonade.Message, 0, len(a.history)+1)
	out = append(out, lemonade.Message{Role: "system", Content: system})
	return append(out, a.history...)
}

// overBudget reports whether the last request came close enough to the context
// window that the next one should be compacted first. Callers must hold a.mu.
func (a *agent) overBudget() bool {
	threshold := a.cfg.contextTokens - a.cfg.maxTokens - ctxSafetyMargin
	if threshold <= 0 {
		return false // misconfigured window; the reactive path still protects us
	}
	return a.lastPrompt > 0 && a.lastPrompt >= threshold
}

// respond runs one conversational turn with a tool-calling loop, then
// synthesizes the final reply. onFiller, if set, is called once with a short
// phrase the moment the agent decides to use tools — so the caller can speak a
// "let me check that" cue while the (slow) tool calls run. Returns reply text,
// audio, and its mime.
func (a *agent) respond(ctx context.Context, userText, ttsFormat string, onFiller func(string)) (string, []byte, string, error) {
	a.turnMu.Lock()
	defer a.turnMu.Unlock()

	a.mu.Lock()
	a.history = append(a.history, lemonade.Message{Role: "user", Content: userText})
	over := a.overBudget()
	a.mu.Unlock()

	// Summarize before we send if the previous turn neared the window, so this
	// turn starts well below it.
	if over {
		a.compact(ctx)
	}

	var tools []lemonade.Tool
	if a.tools != nil {
		tools = a.tools.Tools()
	}

	reply := ""
	announced := false
	for round := 0; ; round++ {
		// After the budget, stop offering tools and nudge the model to answer in
		// words — otherwise it tends to emit one more tool call as plain text,
		// which would be read aloud.
		offer := tools
		nudge := ""
		if round >= a.cfg.maxToolRounds {
			offer = nil
			nudge = finalAnswerNudge
		}

		res, err := a.chat(ctx, offer, nudge)
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
				out, imgs, callErr := a.tools.CallTool(ctx, tc.Name, tc.Arguments)
				if callErr != nil {
					out = "error: " + callErr.Error()
				}
				if strings.TrimSpace(out) == "" {
					if len(imgs) > 0 {
						out = "(image returned)"
					} else {
						out = "(no output)"
					}
				}
				log.Printf("tool %s(%s) -> %d chars, %d image(s)", tc.Name, truncate(tc.Arguments, 100), len(out), len(imgs))
				// Cap the live message: proactive compaction only runs before the
				// loop, so a single runaway page must not blow the window mid-turn.
				out = clampText(out, a.cfg.toolOutputLimit)
				a.mu.Lock()
				a.history = append(a.history, lemonade.Message{Role: "tool", ToolCallID: tc.ID, Content: out, Images: imgs})
				a.mu.Unlock()
			}
			continue
		}

		reply = res.Content
		break
	}

	// Never speak a tool call: the model sometimes emits one as plain text
	// (e.g. when tools were withdrawn at the budget) and llama.cpp returns it as
	// content rather than structured tool_calls.
	if looksLikeToolCall(reply) {
		log.Printf("reply looked like a tool call; suppressing: %s", truncate(oneLine(reply), 120))
		reply = ""
	}
	if reply == "" {
		reply = "Sorry, I couldn't get that just now."
	}
	reply = sanitize(reply)

	a.mu.Lock()
	a.history = append(a.history, lemonade.Message{Role: "assistant", Content: reply})
	a.mu.Unlock()
	a.stripImages() // the model has answered; don't re-send the (large) image next turn

	audio, mime, err := a.tts.SpeakAs(ctx, reply, ttsFormat)
	if err != nil {
		return reply, nil, "", fmt.Errorf("tts: %w", err)
	}
	return reply, audio, mime, nil
}

// chat sends the current conversation. On a context-window overflow it compacts
// the history and retries, so a long conversation degrades gracefully instead
// of wedging the assistant.
func (a *agent) chat(ctx context.Context, offer []lemonade.Tool, nudge string) (*lemonade.ChatResult, error) {
	for attempt := 0; ; attempt++ {
		a.mu.Lock()
		msgs := a.messages()
		a.mu.Unlock()
		// Transient steering for this call only; never persisted to history.
		if nudge != "" {
			msgs = append(msgs, lemonade.Message{Role: "user", Content: nudge})
		}

		res, err := a.lem.Chat(ctx, a.cfg.chatModel, msgs, offer)
		if err == nil {
			a.mu.Lock()
			a.lastPrompt = res.PromptTokens
			a.mu.Unlock()
			return res, nil
		}
		if !errors.Is(err, lemonade.ErrContextLength) || attempt >= maxCompactRetries {
			return nil, err
		}
		if !a.compact(ctx) {
			return nil, err // nothing left to shed
		}
	}
}

// compact folds the oldest turns into the rolling summary and drops them,
// freeing context. It returns true if it shed anything.
func (a *agent) compact(ctx context.Context) bool {
	a.mu.Lock()
	older, _, ok := splitForCompaction(a.history)
	prev := a.summary
	olderLen := len(older)
	a.mu.Unlock()

	// Nothing older than the current turn to evict (e.g. one turn with a huge
	// tool result, or a summary that has itself grown too large): clip in place.
	if !ok {
		return a.truncateOversized()
	}

	newSummary, err := a.summarize(ctx, prev, older)

	a.mu.Lock()
	defer a.mu.Unlock()
	if olderLen > len(a.history) { // history changed under us; bail safely
		return false
	}
	if err != nil {
		// Summarizer unavailable: drop the old turns anyway (lossy but safe).
		a.history = append([]lemonade.Message(nil), a.history[olderLen:]...)
		log.Printf("compact: summarize failed, dropped %d msg(s): %v", olderLen, err)
		return true
	}
	// Cap the summary so it can never grow into the window on its own.
	a.summary = clampText(newSummary, summaryCharLimit)
	a.history = append([]lemonade.Message(nil), a.history[olderLen:]...)
	log.Printf("compact: folded %d msg(s) into summary (%d chars), prompt was %d tok", olderLen, len(a.summary), a.lastPrompt)
	return true
}

// summarize produces an updated rolling summary from the previous summary plus
// the evicted turns. It calls the model directly (never a.chat) so it can't
// recurse into compaction.
func (a *agent) summarize(ctx context.Context, prev string, older []lemonade.Message) (string, error) {
	var b strings.Builder
	if prev != "" {
		b.WriteString("Current summary:\n")
		b.WriteString(prev)
		b.WriteString("\n\n")
	}
	b.WriteString("Conversation to fold in:\n")
	b.WriteString(renderTranscript(older))
	b.WriteString("\nReturn the updated summary.")

	msgs := []lemonade.Message{
		{Role: "system", Content: summarizerPrompt},
		{Role: "user", Content: b.String()},
	}
	res, err := a.lem.Chat(ctx, a.cfg.chatModel, msgs, nil)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(res.Content), nil
}

const summarizerPrompt = "You maintain a brief running memory of a voice-assistant conversation. " +
	"Rewrite it to fold in the new turns, keeping only durable facts, user preferences, decisions, and unresolved questions. " +
	"Drop greetings and small talk. Be concise — a few plain sentences, no markdown or lists."

// splitForCompaction returns the oldest messages to evict and the recent ones
// to keep. It cuts at a user-message boundary so an assistant tool-call and its
// tool results are never separated. ok is false when there is nothing older
// than the current turn to evict.
func splitForCompaction(h []lemonade.Message) (older, keep []lemonade.Message, ok bool) {
	split := -1
	kept := 0
	for i := len(h) - 1; i >= 0; i-- {
		kept++
		if kept >= keepRecent && h[i].Role == "user" {
			split = i
			break
		}
	}
	if split <= 0 {
		// Shorter than the keep window, or no early boundary: evict everything
		// before the last user turn.
		split = lastUserIndex(h)
	}
	if split <= 0 {
		return nil, h, false
	}
	return h[:split], h[split:], true
}

func lastUserIndex(h []lemonade.Message) int {
	for i := len(h) - 1; i >= 0; i-- {
		if h[i].Role == "user" {
			return i
		}
	}
	return -1
}

// truncateOversized clips every history message that exceeds the digest limit
// (and, failing that, the rolling summary) down to that limit. It is the last
// resort when a single turn — or the summary alone — fills the window and there
// is nothing to summarize. Each clip lands exactly at the limit so an already
// clipped message is never reselected: progress is honest and the chat() retry
// loop terminates.
func (a *agent) truncateOversized() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	changed := false
	for i := range a.history {
		if len(a.history[i].Content) > toolDigestLimit {
			a.history[i].Content = clampText(a.history[i].Content, toolDigestLimit)
			changed = true
		}
	}
	if !changed && len(a.summary) > summaryCharLimit {
		a.summary = clampText(a.summary, summaryCharLimit)
		changed = true
	}
	if !changed {
		// Nothing textual left to shed: drop inline images (the heaviest payload)
		// so the retry can fit rather than wedge.
		for i := range a.history {
			if len(a.history[i].Images) > 0 {
				a.history[i].Images = nil
				if strings.TrimSpace(a.history[i].Content) == "" {
					a.history[i].Content = "(image dropped to fit context)"
				}
				changed = true
			}
		}
	}
	return changed
}

// stripImages drops inline images from history once a turn is done. A rendered
// panel is large and only needed while the model forms its reply; keeping it
// would re-send the whole image on every later turn and undo the context bound.
func (a *agent) stripImages() {
	a.mu.Lock()
	defer a.mu.Unlock()
	for i := range a.history {
		if len(a.history[i].Images) > 0 {
			a.history[i].Images = nil
			if strings.TrimSpace(a.history[i].Content) == "" {
				a.history[i].Content = "(image was shown to the assistant)"
			}
		}
	}
}

// clampText trims s to at most limit bytes, marking the cut. The result is
// exactly limit bytes when clipped, so re-clamping the same value is a no-op.
// limit <= 0 disables clamping.
func clampText(s string, limit int) string {
	if limit <= 0 || len(s) <= limit {
		return s
	}
	const marker = " [truncated]"
	if limit <= len(marker) {
		return s[:limit]
	}
	return s[:limit-len(marker)] + marker
}

// renderTranscript flattens turns into plain text for the summarizer, clipping
// bulky tool results so the summary call itself stays within the window.
func renderTranscript(msgs []lemonade.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		switch m.Role {
		case "user":
			b.WriteString("User: ")
			b.WriteString(m.Content)
			b.WriteString("\n")
		case "assistant":
			if c := strings.TrimSpace(m.Content); c != "" {
				b.WriteString("Assistant: ")
				b.WriteString(c)
				b.WriteString("\n")
			}
			for _, tc := range m.ToolCalls {
				b.WriteString("Assistant used ")
				b.WriteString(tc.Name)
				b.WriteString("(")
				b.WriteString(truncate(tc.Arguments, 120))
				b.WriteString(")\n")
			}
		case "tool":
			b.WriteString("Result: ")
			b.WriteString(truncate(oneLine(m.Content), toolDigestLimit))
			b.WriteString("\n")
		}
	}
	return b.String()
}

func oneLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
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
		// ?format= lets a client pick the audio encoding (e.g. the ESP32 asks for
		// raw 16-bit pcm); empty falls back to the configured TTS_FORMAT.
		reply, audio, mime, err := a.respond(ctx, text, r.URL.Query().Get("format"), func(t string) { filler = t })
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
		audio, mime, err := a.tts.SpeakAs(ctx, text, r.URL.Query().Get("format"))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		writeAudio(w, audio, mime)
	})

	// POST /api/stt -> {"text": "..."}. Two body shapes:
	//   - streaming (the ESP32): raw signed-16-bit LE mono PCM, Content-Type
	//     audio/L16, sample rate in ?rate= (default 16000); wrapped to WAV here.
	//   - legacy: multipart/form-data field "file" = a complete WAV.
	mux.HandleFunc("POST /api/stt", func(w http.ResponseWriter, r *http.Request) {
		var wav []byte
		if strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/") {
			f, _, err := r.FormFile("file")
			if err != nil {
				http.Error(w, "expected multipart field 'file' (wav)", http.StatusBadRequest)
				return
			}
			defer f.Close()
			wav, _ = io.ReadAll(f)
		} else {
			rate, _ := strconv.Atoi(r.URL.Query().Get("rate"))
			if rate <= 0 {
				rate = 16000
			}
			pcm, err := io.ReadAll(r.Body) // chunked transfer is de-chunked transparently
			if err != nil || len(pcm) == 0 {
				http.Error(w, "empty PCM body", http.StatusBadRequest)
				return
			}
			wav = pcmToWAV(pcm, rate, 1, 16)
		}
		ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
		defer cancel()
		text, err := a.lem.Transcribe(ctx, a.cfg.sttModel, wav, "en")
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"text": text})
	})

	// POST /api/reset -> clears conversation history.
	mux.HandleFunc("POST /api/reset", func(w http.ResponseWriter, r *http.Request) {
		n := a.reset()
		log.Printf("reset: cleared %d message(s)", n)
		fmt.Fprintln(w, "ok")
	})

	log.Printf("Homa %s listening on %s (LLM=%s, TTS=%s voice=%s)", version, a.cfg.addr, a.cfg.chatModel, a.cfg.ttsModel, a.cfg.ttsVoice)
	if err := http.ListenAndServe(a.cfg.addr, mux); err != nil {
		log.Fatalf("%v", err)
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

// pcmToWAV prepends a 44-byte canonical WAV header to raw PCM samples so the
// transcription endpoint gets a proper audio container.
func pcmToWAV(pcm []byte, rate, channels, bits int) []byte {
	n := len(pcm)
	byteRate := rate * channels * bits / 8
	blockAlign := channels * bits / 8
	h := make([]byte, 44+n)
	copy(h[0:4], "RIFF")
	binary.LittleEndian.PutUint32(h[4:8], uint32(36+n))
	copy(h[8:12], "WAVE")
	copy(h[12:16], "fmt ")
	binary.LittleEndian.PutUint32(h[16:20], 16) // PCM fmt chunk size
	binary.LittleEndian.PutUint16(h[20:22], 1)  // audio format = PCM
	binary.LittleEndian.PutUint16(h[22:24], uint16(channels))
	binary.LittleEndian.PutUint32(h[24:28], uint32(rate))
	binary.LittleEndian.PutUint32(h[28:32], uint32(byteRate))
	binary.LittleEndian.PutUint16(h[32:34], uint16(blockAlign))
	binary.LittleEndian.PutUint16(h[34:36], uint16(bits))
	copy(h[36:40], "data")
	binary.LittleEndian.PutUint32(h[40:44], uint32(n))
	copy(h[44:], pcm)
	return h
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

// looksLikeToolCall reports whether s is a tool-call template the model leaked
// into its reply text (Qwen/Hermes XML, or a bare JSON call) instead of a real
// answer — so it is never read aloud.
func looksLikeToolCall(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	if strings.Contains(s, "<tool_call") || strings.Contains(s, "<function=") || strings.Contains(s, "<function ") {
		return true
	}
	if strings.HasPrefix(s, "{") && strings.Contains(s, `"name"`) && strings.Contains(s, `"arguments"`) {
		return true
	}
	return false
}

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

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
		log.Printf("invalid %s=%q, using %d", key, v, def)
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
