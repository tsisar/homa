// Package tts provides speech synthesis via an OpenAI-compatible
// POST /v1/audio/speech endpoint. This works unchanged against Lemonade's
// Kokoro (model kokoro-v1) and against any other OpenAI-compatible TTS
// server (e.g. Chatterbox-TTS-Server) — only config changes.
package tts

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"

	"github.com/tsisar/extended-log-go/log"
)

// OpenAISpeech calls POST {BaseURL}/audio/speech and returns the audio bytes
// plus the response content-type (so callers can write/play them as-is).
type OpenAISpeech struct {
	BaseURL string  // e.g. http://192.168.88.83:8000/api/v1
	Model   string  // e.g. kokoro-v1
	Voice   string  // e.g. af_heart
	Format  string  // response_format: wav | mp3 | pcm | opus | flac
	Speed   float64 // speech rate; 0 = omit (backend default). StyleTTS2 honors ~0.8–1.3
	HTTP    *http.Client
}

func NewOpenAISpeech(baseURL, model, voice, format string, speed float64) *OpenAISpeech {
	return &OpenAISpeech{
		BaseURL: strings.TrimRight(baseURL, "/"),
		Model:   model,
		Voice:   voice,
		Format:  format,
		Speed:   speed,
		HTTP:    &http.Client{},
	}
}

type speechRequest struct {
	Model          string  `json:"model"`
	Input          string  `json:"input"`
	Voice          string  `json:"voice"`
	ResponseFormat string  `json:"response_format"`
	Speed          float64 `json:"speed,omitempty"` // omitted when 0 → backend default
}

// Speak synthesizes text in the configured default format.
func (s *OpenAISpeech) Speak(ctx context.Context, text string) ([]byte, string, error) {
	return s.SpeakAs(ctx, text, s.Format)
}

// SpeakAs synthesizes text in the given response_format (wav|mp3|pcm|opus|flac);
// an empty format falls back to the configured default. This lets one endpoint
// serve different clients — e.g. wav for browsers, raw 16-bit pcm for the ESP32.
func (s *OpenAISpeech) SpeakAs(ctx context.Context, text, format string) ([]byte, string, error) {
	if format = strings.TrimSpace(format); format == "" {
		format = s.Format
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, "", fmt.Errorf("tts: empty text")
	}

	// The ESP32 plays raw 16-bit PCM, but not every backend emits "pcm"
	// (StyleTTS2 only does wav/mp3). For a pcm request, fetch wav and strip it to
	// raw PCM ourselves — works against any wav-capable backend.
	backendFormat, toPCM := format, format == "pcm"
	if toPCM {
		backendFormat = "wav"
	}

	body, err := json.Marshal(speechRequest{
		Model:          s.Model,
		Input:          text,
		Voice:          s.Voice,
		ResponseFormat: backendFormat,
		Speed:          s.Speed,
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
	log.Tracef("[tts] request: model=%s voice=%s format=%s speed=%.2f input=%dB", s.Model, s.Voice, backendFormat, s.Speed, len(text))

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

	if toPCM {
		pcm, err := wavToPCM16(data)
		if err != nil {
			return nil, "", fmt.Errorf("tts: wav->pcm: %w", err)
		}
		return pcm, "audio/l16", nil
	}

	mime := resp.Header.Get("Content-Type")
	if mime == "" || strings.HasPrefix(mime, "application/json") {
		mime = "audio/" + format
	}
	return data, mime, nil
}

// wavToPCM16 extracts mono 16-bit little-endian PCM from a WAV blob, converting
// 32-bit float samples (e.g. Kokoro's WAV) to int16. Both StyleTTS2 (int16) and
// Kokoro (float32) emit 24 kHz mono, which is what the ESP32 plays.
func wavToPCM16(b []byte) ([]byte, error) {
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a WAV (%d bytes)", len(b))
	}
	var audioFormat, bits uint16
	var data []byte
	for off := 12; off+8 <= len(b); {
		id := string(b[off : off+4])
		sz := int(binary.LittleEndian.Uint32(b[off+4 : off+8]))
		body := off + 8
		end := body + sz
		if end > len(b) {
			end = len(b)
		}
		switch id {
		case "fmt ":
			if body+16 <= len(b) {
				audioFormat = binary.LittleEndian.Uint16(b[body : body+2])
				bits = binary.LittleEndian.Uint16(b[body+14 : body+16])
			}
		case "data":
			data = b[body:end]
		}
		off = body + sz
		if sz%2 == 1 { // chunks are word-aligned
			off++
		}
	}
	if data == nil {
		return nil, fmt.Errorf("no data chunk")
	}
	switch {
	case audioFormat == 1 && bits == 16:
		return data, nil // already 16-bit PCM
	case audioFormat == 3 && bits == 32:
		out := make([]byte, len(data)/4*2)
		for i, j := 0, 0; i+4 <= len(data); i, j = i+4, j+2 {
			f := math.Float32frombits(binary.LittleEndian.Uint32(data[i : i+4]))
			v := int32(f * 32767)
			if v > 32767 {
				v = 32767
			} else if v < -32768 {
				v = -32768
			}
			binary.LittleEndian.PutUint16(out[j:j+2], uint16(int16(v)))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported WAV: format=%d bits=%d", audioFormat, bits)
	}
}

func snippet(b []byte) string {
	const max = 300
	if len(b) > max {
		return string(b[:max]) + "…"
	}
	return string(b)
}
