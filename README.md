# Homa

**Homa** — *Home Orchestration & Monitoring Assistant*. A small Go voice
assistant for the home and homelab: it listens, thinks, searches the web, and
talks back. The brain, ears and default voice run on a local
[Lemonade Server](https://lemonade-server.ai); web search runs as an MCP tool
behind the homelab gateway.

```
you ──speak/text──▶ Homa (Go)
                      │  chat  ─▶ LLM gateway /chat/completions   ─▶ Lemonade (Qwen3.6-35B-A3B, thinking off)
                      │  tools ─▶ MCP gateway (kgateway) ─▶ ddgo-mcp: web_search / web_fetch
                      │  voice ─▶ LLM gateway /audio/speech       ─▶ Lemonade (Kokoro, voice af_heart)
                      ▼
                   audio reply  ("Let me look that up." → … → answer)

mic path (ESP32 client): wav ─▶ LLM gateway /audio/transcriptions ─▶ Lemonade (Whisper) ─▶ text
```

Chat, STT and TTS all share **one** endpoint (`API_URL`) — the LLM gateway's
`local` provider, which fronts Lemonade for both chat and audio.

The LLM, STT and TTS are reached over the OpenAI-compatible API, so swapping a
model or the TTS server (e.g. Chatterbox on a Mac for a more expressive voice)
is a config change, not a code change.

## Prerequisites

- Go 1.25+
- macOS (`afplay` is used for local playback)
- A reachable Lemonade Server with these models pulled:
    - chat: `Qwen3.6-35B-A3B-MTP-GGUF` (default; any chat model works)
    - TTS: `kokoro-v1`
    - STT: `Whisper-Large-v3-Turbo`
    - reached via `API_URL` — the LLM gateway (`http://llm.tsisar.local/local/v1`,
      default) or Lemonade directly (`http://192.168.88.83:8000/api/v1`)

No local setup needed for English — Kokoro runs on the Lemonade box.

## Run

```sh
# one-shot: ask, print + play the spoken reply
go run . -once "Who are you?"

# speak arbitrary text locally (no LLM)
go run . -say "Testing the voice."

# with web search enabled
MCP_URL=http://mcp.tsisar.local/ MCP_ALLOW="web_*" go run . -once "What's the weather in Kyiv?"

# HTTP server (this is what the ESP32 will talk to)
go run .
```

### HTTP endpoints

| Method & path     | Body                       | Response                                               |
|-------------------|----------------------------|--------------------------------------------------------|
| `GET  /healthz`   | —                          | `ok`                                                   |
| `POST /api/talk`  | `{"text":"..."}`           | audio reply; `X-Reply-Text` (+ `X-Filler-Text`) header |
| `POST /api/say`   | `{"text":"..."}`           | audio (TTS only)                                       |
| `POST /api/stt`   | multipart `file` = PCM WAV | `{"text":"..."}` (Whisper)                             |
| `POST /api/reset` | —                          | clears conversation history                            |

```sh
curl -X POST localhost:8080/api/talk -d '{"text":"Tell me a joke"}' -o reply.wav -D - | grep -i x-reply
afplay reply.wav
```

## Configuration (env vars)

| Var                | Default                                                                                                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `API_URL`          | `http://llm.tsisar.local/local/v1` — single endpoint for chat + STT + TTS (LLM gateway, local provider). Set to `http://192.168.88.83:8000/api/v1` to hit Lemonade directly |
| `CHAT_MODEL`       | `Qwen3.6-35B-A3B-MTP-GGUF`                                                                                                                                                  |
| `STT_MODEL`        | `Whisper-Large-v3-Turbo`                                                                                                                                                    |
| `TTS_URL`          | = `API_URL` (optional override, e.g. Chatterbox on a Mac)                                                                                                                   |
| `TTS_MODEL`        | `kokoro-v1`                                                                                                                                                                 |
| `TTS_VOICE`        | `af_heart` (warm female; `am_michael`, `bf_emma`)                                                                                                                           |
| `TTS_FORMAT`       | `wav`                                                                                                                                                                       |
| `DISABLE_THINKING` | `true`                                                                                                                                                                      |
| `MCP_URL`          | empty (MCP disabled); e.g. `http://mcp.tsisar.local/`                                                                                                                       |
| `MCP_ALLOW`        | empty (no tools); CSV of names/globs, e.g. `web_*` or `*`                                                                                                                   |
| `SEARCH_FILLER`    | `Let me look that up.` — spoken once when tools start; empty disables                                                                                                       |
| `CONTEXT_TOKENS`   | `4096` — model context window; **must match the backend** (`llama-server -c`). History is summarized before it fills                                                        |
| `MAX_TOKENS`       | `512` — reply budget reserved below `CONTEXT_TOKENS`                                                                                                                        |
| `ADDR`             | `:8080`                                                                                                                                                                     |
| `SYSTEM_PROMPT`    | English, voice-optimized; sets the Homa persona                                                                                                                             |
| `LOG_LEVEL`        | `trace` (unset → trace) — `trace` logs full LLM requests/responses; `debug`/`info`/`warn`/`error` to quiet down                                                             |
| `LOG_TIMEZONE`     | host local; e.g. `Asia/Dubai` — timestamp zone for logs                                                                                                                     |

### Switching to Chatterbox (expressive) later

Run Chatterbox-TTS-Server (OpenAI-compatible) on the Mac and point Homa at it —
no code change:

```sh
TTS_URL=http://localhost:8004/v1 TTS_MODEL=chatterbox TTS_VOICE=<voice> go run .
```

## Web search via MCP

Homa can call tools exposed by an MCP server over **Streamable HTTP**. Point it
at one with `MCP_URL` and pick which tools the model may see with `MCP_ALLOW`.

Search is provided by **[ddgo-mcp](https://github.com/tsisar/ddgo-mcp)** — a tiny
Go MCP server (`search` + `fetch`; DuckDuckGo + readability, no API key, no
browser) deployed in the k3s homelab behind the kgateway/agentgateway MCP
gateway as target `web`. The gateway federates several MCP backends behind one
endpoint and namespaces tools as `<target>_<tool>`, so ddgo-mcp's tools appear
as `web_search` / `web_fetch` alongside `grafana_*`, `postgres_*`, etc.

```sh
MCP_URL=http://mcp.tsisar.local/ MCP_ALLOW="web_*" go run .
```

### Read-only Grafana tools

Grafana tools can be added the same way. Since `matchAllow` only does trailing
`prefix*` globs and Grafana namespaces every **read** tool under a verb prefix
(`get_`, `list_`, `query_`, `search_`, `find_`), a read-only set is expressible
without naming each tool — and it excludes every mutating tool (`create_*`,
`update_*`, `add_*`, `alerting_manage_*`) plus the raw API passthrough
`grafana_grafana_api_request`:

```sh
MCP_URL=http://mcp.tsisar.local/ \
  MCP_ALLOW="web_*,grafana_get_*,grafana_list_*,grafana_query_*,grafana_search_*,grafana_find_*,grafana_generate_deeplink" \
  go run .
```

That exposes ~45 tools, though — a lot of tool-schema to send every turn and
hard for a small local model to choose between. For voice, prefer a tight,
curated subset, e.g.:

```sh
MCP_ALLOW="web_*,grafana_query_prometheus,grafana_query_loki_logs,grafana_find_error_pattern_logs,grafana_list_alert_groups,grafana_get_current_oncall_users,grafana_search_dashboards"
```

How it works:

- `respond()` runs the tool-use loop: send `tools` → on `tool_calls`, call each
  via `internal/mcp` and feed back a `role:"tool"` message → repeat (≤5 rounds,
  then answer without tools).
- When Homa first decides to use a tool it speaks a short filler (`SEARCH_FILLER`,
  e.g. "Let me look that up.") to mask tool latency. The CLI plays it immediately;
  `/api/talk` returns it in the `X-Filler-Text` header. (On the ESP32 it becomes
  a first audio segment over the future WebSocket.)
- **Images:** a tool result that returns a picture (e.g. `grafana_get_panel_image`)
  is passed to the model as an inline `image_url` part, so a vision-capable chat
  model can read it — Homa can answer "what does this panel show" off a rendered
  Grafana panel. The image is kept only for the turn that fetched it and then
  stripped from history (a panel PNG is large; re-sending it every turn would
  blow the context). Requires a vision model served with multimodal support; a
  text-only model will just ignore the image.
- **Security:** `MCP_ALLOW` is an explicit allow-list (exact names or a trailing
  `prefix*` glob — no other wildcards; `*` = everything); empty = no tools. Keep
  it tight — the gateway also fronts tools a voice assistant should never reach:
  `postgres_execute_sql` (arbitrary SQL), `grafana_grafana_api_request` (raw
  Grafana API — can write/delete), and Grafana `create_*` / `update_*` writes.

## Notes

- **Routing:** chat, STT and TTS all go through the kgateway LLM gateway via a
  single `API_URL` (`local` provider — Homa uses only local models, never the
  gateway's OpenAI / Anthropic providers). Chat hits the `local` AI backend;
  `/audio/*` rides a static passthrough backend on the same gateway (the AI
  backend only speaks chat-completions and 503s on audio). The gateway preserves
  `chat_template_kwargs` (thinking-disable) and tool calls. Point `API_URL`
  straight at Lemonade to bypass the gateway entirely.
- `Qwen3.6-35B-A3B` is a reasoning model: it emits chain-of-thought in
  `reasoning_content` and leaves `content` empty unless thinking is disabled.
  Homa sends `chat_template_kwargs.enable_thinking=false` (toggle with
  `DISABLE_THINKING`). Without it the reply is empty.
- Kokoro audio is 24000 Hz mono; Lemonade returns it as float WAV, passed
  through to clients / `afplay` as-is.
- **Context management:** the conversation is sent in full every turn, so it
  grows until it hits the model's window. Homa keeps the system prompt plus the
  recent turns verbatim and folds older turns into a rolling summary. It compacts
  proactively (when the previous request's reported `prompt_tokens` nears
  `CONTEXT_TOKENS - MAX_TOKENS`) and reactively (on a `context_length_exceeded`
  from the backend it summarizes and retries), so the assistant never wedges on a
  long conversation. `POST /api/reset` clears both history and summary.

## Parked: Ukrainian

Ukrainian voice is on hold (Kokoro has no Ukrainian; the Piper `uk_UA` voice had
poor prosody). `scripts/setup.sh`, `.venv/`, and `voices/` are leftovers from
that path and are unused by the English build.

## Next: ESP32 client

Target device: Waveshare ESP32-C6 Touch AMOLED 2.16 (ES8311 codec, dual mic,
speaker). Planned flow: push-to-talk on the touchscreen → record mic via I2S →
`POST /api/stt` → `POST /api/talk` → play the returned audio through the speaker.
"Homa" becomes the wake word and a single WebSocket (streaming the filler then
the answer) is the later upgrade.
