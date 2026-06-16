package tts

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// wav builds a minimal mono WAV: audioFormat 1 (PCM) or 3 (float), given bits.
func wav(audioFormat, bits uint16, rate uint32, data []byte) []byte {
	b := &bytes.Buffer{}
	b.WriteString("RIFF")
	binary.Write(b, binary.LittleEndian, uint32(36+len(data)))
	b.WriteString("WAVE")
	b.WriteString("fmt ")
	binary.Write(b, binary.LittleEndian, uint32(16))
	binary.Write(b, binary.LittleEndian, audioFormat)
	binary.Write(b, binary.LittleEndian, uint16(1)) // mono
	binary.Write(b, binary.LittleEndian, rate)
	binary.Write(b, binary.LittleEndian, rate*uint32(bits/8))
	binary.Write(b, binary.LittleEndian, uint16(bits/8))
	binary.Write(b, binary.LittleEndian, bits)
	b.WriteString("data")
	binary.Write(b, binary.LittleEndian, uint32(len(data)))
	b.Write(data)
	return b.Bytes()
}

func le16(samples ...int16) []byte {
	out := make([]byte, len(samples)*2)
	for i, s := range samples {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(s))
	}
	return out
}

func TestWavToPCM16_Int16Passthrough(t *testing.T) {
	pcm := le16(0, 1, -1, 32767, -32768)
	got, err := wavToPCM16(wav(1, 16, 24000, pcm))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, pcm) {
		t.Errorf("int16 PCM not passed through unchanged:\n got %v\nwant %v", got, pcm)
	}
}

func TestWavToPCM16_Float32(t *testing.T) {
	floats := []float32{0, 1.0, -1.0, 0.5}
	data := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(f))
	}
	got, err := wavToPCM16(wav(3, 32, 24000, data))
	if err != nil {
		t.Fatal(err)
	}
	want := le16(0, 32767, -32767, 16383) // 1.0*32767, -1.0*32767, 0.5*32767
	if !bytes.Equal(got, want) {
		t.Errorf("float32->int16 wrong:\n got %v\nwant %v", got, want)
	}
}

func TestWavToPCM16_Bad(t *testing.T) {
	if _, err := wavToPCM16([]byte("not a wav")); err == nil {
		t.Error("expected error for non-WAV input")
	}
	if _, err := wavToPCM16(wav(7, 24, 24000, []byte{1, 2, 3})); err == nil {
		t.Error("expected error for unsupported WAV subtype")
	}
}
