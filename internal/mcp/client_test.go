package mcp

import (
	"strings"
	"testing"
	"time"
)

func TestNormalizeMode(t *testing.T) {
	tests := map[string]string{
		"auto":  ReconnectAuto,
		"ask":   ReconnectAsk,
		"off":   ReconnectOff,
		"ASK":   ReconnectAsk,  // case-insensitive
		" off ": ReconnectOff,  // trimmed
		"":      ReconnectAuto, // unset → auto (prior behavior)
		"bogus": ReconnectAuto, // unknown → auto, never a broken mode
	}
	for in, want := range tests {
		if got := normalizeMode(in); got != want {
			t.Errorf("normalizeMode(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestExecutorBackoff(t *testing.T) {
	e := &Executor{}
	tests := []struct {
		fails int
		want  time.Duration
	}{
		{0, 1 * time.Second},
		{1, 2 * time.Second},
		{2, 4 * time.Second},
		{3, 8 * time.Second},
		{4, 16 * time.Second},
		{5, 30 * time.Second}, // 32s capped to 30s
		{6, 30 * time.Second},
		{100, 30 * time.Second}, // no overflow at large failure counts
	}
	for _, tt := range tests {
		e.dialFails = tt.fails
		if got := e.backoff(); got != tt.want {
			t.Errorf("backoff(dialFails=%d) = %s, want %s", tt.fails, got, tt.want)
		}
	}
}

// the exact grafana_list_datasources response shape from the live gateway
const realDatasources = `{"datasources":[` +
	`{"id":4,"uid":"influxdb","name":"InfluxDB","type":"influxdb","isDefault":false},` +
	`{"id":2,"uid":"P8E80F9AEF21F6940","name":"Loki","type":"loki","isDefault":false},` +
	`{"id":1,"uid":"PBFA97CFB590B2093","name":"Prometheus","type":"prometheus","isDefault":true},` +
	`{"id":3,"uid":"P214B5B846CF3925F","name":"Tempo","type":"tempo","isDefault":false}` +
	`],"total":4,"hasMore":false}`

func TestDatasourceFix(t *testing.T) {
	dc := parseDatasources(realDatasources)
	if dc == nil {
		t.Fatal("parseDatasources returned nil for valid input")
	}

	tests := []struct {
		name string // tool name
		key  string // arg key
		in   string
		want string // expected value after fix
	}{
		{"grafana_query_prometheus", "datasourceUid", "prometheus", "PBFA97CFB590B2093"},
		{"grafana_query_prometheus", "datasourceUid", "Prometheus", "PBFA97CFB590B2093"},
		{"grafana_query_loki_logs", "datasourceUid", "loki", "P8E80F9AEF21F6940"},
		{"grafana_query_prometheus", "datasourceUid", "default", "PBFA97CFB590B2093"},
		{"grafana_query_prometheus", "datasourceUid", "PBFA97CFB590B2093", "PBFA97CFB590B2093"}, // already correct
		{"grafana_query_prometheus", "datasourceUid", "pbfa97cfb590b2093", "PBFA97CFB590B2093"}, // wrong case
		{"grafana_query_prometheus", "datasourceUid", "bogus", "bogus"},                         // unknown: untouched
		{"grafana_query_pyroscope", "data_source_uid", "influxdb", "influxdb"},                  // name==type==uid
		{"web_search", "datasourceUid", "prometheus", "prometheus"},                             // non-grafana: untouched
		// the generic uid key (dashboard id) must NOT be resolved
		{"grafana_get_dashboard_summary", "uid", "loki", "loki"},
	}
	for _, tt := range tests {
		args := map[string]any{tt.key: tt.in}
		dc.fix(tt.name, args)
		if got := args[tt.key]; got != tt.want {
			t.Errorf("fix(%s,{%s:%q}) => %v, want %q", tt.name, tt.key, tt.in, got, tt.want)
		}
	}
}

func TestDatasourceFixNilSafe(t *testing.T) {
	var dc *dsCache // nil
	args := map[string]any{"datasourceUid": "prometheus"}
	dc.fix("grafana_query_prometheus", args) // must not panic
	if args["datasourceUid"] != "prometheus" {
		t.Errorf("nil cache must leave args untouched")
	}
}

func TestParseDatasourcesAmbiguousType(t *testing.T) {
	// two prometheus datasources: the shared "prometheus" type key is ambiguous
	twoDefault := `{"datasources":[` +
		`{"uid":"A","name":"Prom A","type":"prometheus","isDefault":true},` +
		`{"uid":"B","name":"Prom B","type":"prometheus","isDefault":false}]}`
	dc := parseDatasources(twoDefault)
	if dc == nil {
		t.Fatal("nil cache")
	}
	if got := dc.byKey["prometheus"]; got != "A" { // unique default breaks the tie
		t.Errorf("ambiguous type with one default => %q, want A", got)
	}
	if dc.byKey["prom a"] != "A" || dc.byKey["prom b"] != "B" { // unique names still resolve
		t.Errorf("unique names failed: %v", dc.byKey)
	}

	none := `{"datasources":[` +
		`{"uid":"A","name":"Prom A","type":"prometheus","isDefault":false},` +
		`{"uid":"B","name":"Prom B","type":"prometheus","isDefault":false}]}`
	dc2 := parseDatasources(none)
	if _, ok := dc2.byKey["prometheus"]; ok { // truly ambiguous => skipped, not guessed
		t.Errorf("ambiguous type with no default must be skipped, got %q", dc2.byKey["prometheus"])
	}
}

func TestParseDatasourcesBadInput(t *testing.T) {
	for _, in := range []string{"", "not json", `{"datasources":[]}`, `{}`} {
		if dc := parseDatasources(in); dc != nil {
			t.Errorf("parseDatasources(%q) = %v, want nil", in, dc)
		}
	}
}

func TestDatasourceHint(t *testing.T) {
	e := &Executor{ds: parseDatasources(realDatasources)}
	h := e.DatasourceHint()
	for _, want := range []string{"Prometheus", "Loki", "default", "datasourceUid", "grafana_list_prometheus_metric_names"} {
		if !strings.Contains(h, want) {
			t.Errorf("DatasourceHint missing %q:\n%s", want, h)
		}
	}
	// no datasources -> empty hint
	if (&Executor{}).DatasourceHint() != "" {
		t.Error("empty executor should produce no hint")
	}
}
