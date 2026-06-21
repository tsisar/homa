package lemonade

import (
	"encoding/json"
	"testing"
)

// TestFlexCodeDecode reproduces the crash where llama-server returns a 500 whose
// body carries a numeric "code" (e.g. {"error":{"code":500}}). Before the fix the
// whole response failed to decode (json: cannot unmarshal number into ... string),
// masking the real upstream error.
func TestFlexCodeDecode(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{"numeric code (llama-server 500)", `{"error":{"code":500,"message":"boom","type":"server_error"}}`, "500"},
		{"string code", `{"error":{"code":"context_length_exceeded","message":"too long"}}`, "context_length_exceeded"},
		{"null code", `{"error":{"code":null,"message":"x"}}`, ""},
		{"missing code", `{"error":{"message":"x"}}`, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cr chatResponse
			if err := json.Unmarshal([]byte(tt.body), &cr); err != nil {
				t.Fatalf("unmarshal failed (the original bug): %v", err)
			}
			if cr.Error == nil {
				t.Fatal("cr.Error = nil, want decoded error")
			}
			if got := string(cr.Error.Code); got != tt.want {
				t.Errorf("Code = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestFlexCodeStringComparison confirms the existing string comparison still
// works against the flexCode type (it underlies a Go string).
func TestFlexCodeStringComparison(t *testing.T) {
	var cr chatResponse
	if err := json.Unmarshal([]byte(`{"error":{"code":"context_length_exceeded"}}`), &cr); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if cr.Error.Code != "context_length_exceeded" {
		t.Errorf("comparison broke: Code = %q", cr.Error.Code)
	}
}
