package itofin

import (
	"math"
	"testing"
)

func TestImpliedForward(t *testing.T) {
	quotes := []OptionQuotePair{
		{Strike: 4900.0, CallBid: 240.0, CallAsk: 241.0, PutBid: 15.0, PutAsk: 16.0},
		{Strike: 5000.0, CallBid: 150.0, CallAsk: 151.0, PutBid: 25.0, PutAsk: 26.0},
		{Strike: 5100.0, CallBid: 70.0, CallAsk: 71.0, PutBid: 45.0, PutAsk: 46.0},
		{Strike: 5200.0, CallBid: 20.0, CallAsk: 21.0, PutBid: 95.0, PutAsk: 96.0},
	}
	res, err := ImpliedForward(quotes, 0.25, 0, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.IsValid {
		t.Fatalf("expected valid result, got: %s", res.Status)
	}
	if res.Forward <= 0 {
		t.Fatalf("invalid forward: %v", res.Forward)
	}
	_ = quotes[0].Repr()
	_ = quotes[0].IsValid()
	_ = res.Repr()
	_ = NewOptionQuotePair(5000, 10, 11, 10, 11)
}

func TestForwardStatusDecodesEveryFFICode(t *testing.T) {
	// Codes are those of itofin_implied_forward_calculate (forward_api.rs).
	cases := []struct {
		code  int32
		name  string
		valid bool
	}{
		{0, "Ok", true},
		{1, "WarningCrossedSyntheticQuotes", false},
		{2, "WarningBoxSpreadArbitrage", false},
		{3, "WarningImplausibleDiscountFactor", false},
		{4, "WarningSpotDeviationExceeded", false},
		{5, "DegradedAmericanBounds", false},
		{99, "Unknown(99)", false},
		{-1, "Unknown(-1)", false},
	}
	for _, c := range cases {
		name, valid := forwardStatus(c.code)
		if name != c.name || valid != c.valid {
			t.Errorf("code %d: got (%q, valid=%v), want (%q, valid=%v)", c.code, name, valid, c.name, c.valid)
		}
	}
}

func TestForwardCarryRateIsNilWhenNotFinite(t *testing.T) {
	if got := forwardCarryRate(math.NaN()); got != nil {
		t.Errorf("NaN carry must be nil, got %v", *got)
	}
	if got := forwardCarryRate(math.Inf(1)); got == nil || !math.IsInf(*got, 1) {
		t.Errorf("a computed infinite carry must be passed through, got %v", got)
	}
	got := forwardCarryRate(0.0123)
	if got == nil || *got != 0.0123 {
		t.Errorf("finite carry must round-trip, got %v", got)
	}
}

// linearQuotes builds parity-consistent quotes C - P = d*(f-k) with a +/-half spread.
func linearQuotes(f, d, half float64, strikes []float64) []OptionQuotePair {
	quotes := make([]OptionQuotePair, 0, len(strikes))
	for _, k := range strikes {
		diff := d * (f - k)
		c := math.Max(diff, 0) + 20
		p := c - diff
		quotes = append(quotes, OptionQuotePair{
			Strike: k, CallBid: c - half, CallAsk: c + half, PutBid: p - half, PutAsk: p + half,
		})
	}
	return quotes
}

func TestImpliedForwardWarningStatusesThroughFFI(t *testing.T) {
	strikes := []float64{4900, 5000, 5100, 5200}

	// D = 0.3 is outside the plausible interval [0.5, 1.25] -> code 3.
	res, err := ImpliedForward(linearQuotes(5000, 0.3, 0.5, strikes), 0.25, 0, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Status != "WarningImplausibleDiscountFactor" || res.IsValid {
		t.Errorf("implausible D: status=%q valid=%v", res.Status, res.IsValid)
	}
	// The fit falls back to D = 1, where this book is crossed; the status keeps the earlier warning.
	if !res.Diagnostics.IsCrossed || res.ForwardBidStrict <= res.ForwardAskStrict {
		t.Errorf("implausible D: crossed=%v strict bid=%v ask=%v, want a crossed book",
			res.Diagnostics.IsCrossed, res.ForwardBidStrict, res.ForwardAskStrict)
	}

	// Forward 5000 against spot 3000 deviates by far more than 10% -> code 4.
	spot := 3000.0
	df := 1.0
	res, err = ImpliedForward(linearQuotes(5000, 1.0, 0.5, strikes), 0.25, 0, &spot, &df)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Status != "WarningSpotDeviationExceeded" || res.IsValid || res.Diagnostics.IsCrossed {
		t.Errorf("spot deviation: status=%q valid=%v crossed=%v", res.Status, res.IsValid, res.Diagnostics.IsCrossed)
	}
}

func TestImpliedForwardCrossedQuotesFlagIsCrossed(t *testing.T) {
	// The crossed-synthetic-quote case of the Rust core test.
	quotes := []OptionQuotePair{
		{Strike: 5000, CallBid: 100, CallAsk: 101, PutBid: 80, PutAsk: 81},
		{Strike: 5050, CallBid: 95, CallAsk: 95.5, PutBid: 50, PutAsk: 50.5},
		{Strike: 5100, CallBid: 50, CallAsk: 51, PutBid: 100, PutAsk: 101},
	}
	df := 1.0
	res, err := ImpliedForward(quotes, 0.1, 0, nil, &df)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Status != "WarningCrossedSyntheticQuotes" {
		t.Errorf("status=%q, want WarningCrossedSyntheticQuotes", res.Status)
	}
	if !res.Diagnostics.IsCrossed {
		t.Error("IsCrossed must be true")
	}
	if res.ForwardBidStrict <= res.ForwardAskStrict {
		t.Errorf("strict bid %v must exceed strict ask %v", res.ForwardBidStrict, res.ForwardAskStrict)
	}
}

func TestImpliedForwardConventionCodes(t *testing.T) {
	// Code 1 is the crypto-inverse convention (coin-numeraire parity), whatever the old doc said.
	const trueForward, dCoin = 70000.0, 0.95
	strikes := []float64{65000, 68000, 70000, 72000, 75000}
	inverse := make([]OptionQuotePair, 0, len(strikes))
	for _, k := range strikes {
		diff := dCoin * (1 - k/trueForward)
		c := 0.1 + math.Max(diff, 0)
		p := c - diff
		inverse = append(inverse, OptionQuotePair{Strike: k, CallBid: c - 0.001, CallAsk: c + 0.001, PutBid: p - 0.001, PutAsk: p + 0.001})
	}
	spot := 69500.0
	res, err := ImpliedForward(inverse, 0.25, 1, &spot, nil)
	if err != nil {
		t.Fatalf("crypto convention: %v", err)
	}
	if !res.IsValid || math.Abs(res.Forward-trueForward) > 5 {
		t.Errorf("crypto convention (1): forward=%v status=%q, want ~%v", res.Forward, res.Status, trueForward)
	}
	if want := math.Log(res.Forward/spot) / 0.25; res.ImpliedCarryRate == nil || math.Abs(*res.ImpliedCarryRate-want) > 1e-12 {
		t.Errorf("crypto carry with spot: got %v, want ln(F/S)/T = %v", res.ImpliedCarryRate, want)
	}
	// Without spot the crypto convention has no carry: the core sends NaN, which must surface as nil.
	res, err = ImpliedForward(inverse, 0.25, 1, nil, nil)
	if err != nil {
		t.Fatalf("crypto convention without spot: %v", err)
	}
	if res.ImpliedCarryRate != nil {
		t.Errorf("crypto carry without spot must be nil, got %v", *res.ImpliedCarryRate)
	}

	// Code 2 is the American equity convention and requires spot.
	american := linearQuotes(101, 1.0, 0.05, []float64{95, 97.5, 100, 102.5, 105})
	if _, err := ImpliedForward(american, 0.25, 2, nil, nil); err == nil {
		t.Error("American convention (2) without spot must fail")
	}
	s, d := 100.0, 1.0
	res, err = ImpliedForward(american, 0.25, 2, &s, &d)
	if err != nil {
		t.Fatalf("American convention: %v", err)
	}
	if !res.IsValid || math.Abs(res.Forward-101) > 0.5 {
		t.Errorf("American convention (2): forward=%v status=%q, want ~101", res.Forward, res.Status)
	}
}
