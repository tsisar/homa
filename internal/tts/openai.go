// Package tts provides speech synthesis via an OpenAI-compatible
// POST /v1/audio/speech endpoint. This works unchanged against Lemonade's
// Kokoro (model kokoro-v1) and against any other OpenAI-compatible TTS
// server (e.g. Chatterbox-TTS-Server) — only config changes.
package tts

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/tsisar/extended-log-go/log"
)

// OpenAISpeech calls POST {BaseURL}/audio/speech and returns the audio bytes
// plus the response content-type (so callers can write/play them as-is).
type OpenAISpeech struct {
	BaseURL string // e.g. http://192.168.88.83:8000/api/v1
	Model   string // e.g. kokoro-v1
	Voice   string // e.g. af_heart
	Format  string // response_format: wav | mp3 | pcm | opus | flac
	HTTP    *http.Client
}

func NewOpenAISpeech(baseURL, model, voice, format string) *OpenAISpeech {
	return &OpenAISpeech{
		BaseURL: strings.TrimRight(baseURL, "/"),
		Model:   model,
		Voice:   voice,
		Format:  format,
		HTTP:    &http.Client{},
	}
}

type speechRequest struct {
	Model          string `json:"model"`
	Input          string `json:"input"`
	Voice          string `json:"voice"`
	ResponseFormat string `json:"response_format"`
}

// Speak synthesizes text and returns (audioBytes, contentType).
func (s *OpenAISpeech) Speak(ctx context.Context, text string) ([]byte, string, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, "", fmt.Errorf("tts: empty text")
	}

	body, err := json.Marshal(speechRequest{
		Model:          s.Model,
		Input:          text,
		Voice:          s.Voice,
		ResponseFormat: s.Format,
	})
	if err != nil {
		return nil, "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.BaseURL+"/audio/speech", bytes.NewReader(body))
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("Content-Type", "application/json")

	// Trace: TTS request metadata only — skip the synthesized audio (response is a blob).
	log.Tracef("[tts] request: model=%s voice=%s format=%s input=%dB", s.Model, s.Voice, s.Format, len(text))

	resp, err := s.HTTP.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("speech request: %w", err)
	}
	defer resp.Body.Close()

	data, _ := io.ReadAll(resp.Body)
	// Trace: TTS response — size + status only, never the raw audio bytes.
	log.Tracef("[tts] response: http=%d audio=%dB", resp.StatusCode, len(data))
	if resp.StatusCode != http.StatusOK {
		// Error responses are JSON; surface them rather than the raw audio path.
		return nil, "", fmt.Errorf("tts http %d: %s", resp.StatusCode, snippet(data))
	}

	mime := resp.Header.Get("Content-Type")
	if mime == "" || strings.HasPrefix(mime, "application/json") {
		mime = "audio/" + s.Format
	}
	return data, mime, nil
}

func snippet(b []byte) string {
	const max = 300
	if len(b) > max {
		return string(b[:max]) + "…"
	}
	return string(b)
}
