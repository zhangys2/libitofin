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
		{6, "WarningImplausibleCarry", false},
		{7, "WarningModelFallback", false},
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

func TestForwardConventionAndLevelConstants(t *testing.T) {
	if ForwardEuropeanVanilla != 0 || ForwardCryptoInverse != 1 || ForwardAmericanEquity != 2 {
		t.Errorf("convention codes changed: %d %d %d", ForwardEuropeanVanilla, ForwardCryptoInverse, ForwardAmericanEquity)
	}
	if ForwardLevelAuto != 0 || ForwardLevelFast != 1 || ForwardLevelAccurate != 2 {
		t.Errorf("level codes changed: %d %d %d", ForwardLevelAuto, ForwardLevelFast, ForwardLevelAccurate)
	}
}

func TestForwardTierAndFlagNames(t *testing.T) {
	tiers := map[int32]string{-1: "", 0: "corridor", 1: "model_fast", 2: "model_accurate", 9: "unknown(9)"}
	for code, want := range tiers {
		if got := forwardTier(code); got != want {
			t.Errorf("tier %d: got %q, want %q", code, got, want)
		}
	}
	if got := forwardFlags(0); len(got) != 0 {
		t.Errorf("no flags: got %v", got)
	}
	if got := forwardFlags(1); len(got) != 1 || got[0] != "schedule_ignored" {
		t.Errorf("flag 1: got %v", got)
	}
	got := forwardFlags(7)
	want := []string{"schedule_ignored", "latency_fallback", "non_converged"}
	if len(got) != len(want) {
		t.Fatalf("flags 7: got %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("flags 7: got %v, want %v", got, want)
		}
	}
	if got := forwardFlags(8); len(got) != 1 || got[0] != "pairs_dropped" {
		t.Errorf("flag 8: got %v", got)
	}
	if got := forwardFlags(16); len(got) != 1 || got[0] != "unknown(0x10)" {
		t.Errorf("unknown bit must be surfaced, got %v", got)
	}
}

func TestForwardModelFailureNames(t *testing.T) {
	want := map[uint32]string{
		0: "", 1: "invalid_discount_factor", 2: "insufficient_pairs", 3: "dividends_exceed_spot",
		4: "pricer_unavailable", 5: "no_implied_volatility", 6: "carry_not_bracketed",
		7: "carry_solve_failed", 8: "too_few_strikes", 9: "parity_fit_failed", 10: "unknown(10)",
	}
	for code, name := range want {
		if got := forwardModelFailure(code); got != name {
			t.Errorf("model failure %d: got %q, want %q", code, got, name)
		}
	}
}

func TestImpliedForwardWithOptionsReportsWhyTheModelFailed(t *testing.T) {
	spot, df := 100.0, math.Exp(-0.01*0.5)
	// The quotes carry about 2% a year; the search only looks above 20%.
	opts := &ImpliedForwardOptions{
		CarrySearchRange: &[2]float64{0.20, 0.25},
		CarryFlagRange:   &[2]float64{0.21, 0.24},
	}
	res, err := ImpliedForwardWithOptions(blackQuotes(0.3), 0.5, ForwardAmericanEquity, &spot, &df, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Tier != "corridor" || res.ModelFailure != "carry_not_bracketed" || res.IsValid {
		t.Errorf("tier=%q failure=%q valid=%v", res.Tier, res.ModelFailure, res.IsValid)
	}
	if len(res.Flags) != 1 || res.Flags[0] != "non_converged" {
		t.Errorf("flags=%v", res.Flags)
	}

	clean, err := ImpliedForwardWithOptions(blackQuotes(0.3), 0.5, ForwardAmericanEquity, &spot, &df, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if clean.ModelFailure != "" {
		t.Errorf("a clean model run reported %q", clean.ModelFailure)
	}
}

// blackQuotes builds Black-priced quotes (spot 100, r 1%, q 2%, 0.5y, vol 25%) with a half spread.
func blackQuotes(half float64) []OptionQuotePair {
	const spot, rate, carry, expiry, vol = 100.0, 0.01, 0.02, 0.5, 0.25
	forward := spot * math.Exp((rate-carry)*expiry)
	df := math.Exp(-rate * expiry)
	sd := vol * math.Sqrt(expiry)
	cdf := func(x float64) float64 { return 0.5 * math.Erfc(-x/math.Sqrt2) }
	var quotes []OptionQuotePair
	for _, k := range []float64{92.5, 95, 97.5, 100, 102.5, 105, 107.5} {
		d1 := (math.Log(forward/k) + 0.5*sd*sd) / sd
		d2 := d1 - sd
		call := df * (forward*cdf(d1) - k*cdf(d2))
		put := df * (k*cdf(-d2) - forward*cdf(-d1))
		quotes = append(quotes, OptionQuotePair{Strike: k, CallBid: call - half, CallAsk: call + half, PutBid: put - half, PutAsk: put + half})
	}
	return quotes
}

func TestImpliedForwardWithOptionsReportsTheModelTier(t *testing.T) {
	spot, df := 100.0, math.Exp(-0.01*0.5)
	// European Black prices are not American prices, so a tight spread would cross the synthetic
	// bounds; a 0.3 half spread keeps the estimate clean.
	res, err := ImpliedForwardWithOptions(blackQuotes(0.3), 0.5, ForwardAmericanEquity, &spot, &df, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Tier != "model_fast" || len(res.Flags) != 0 || res.Status != "Ok" || !res.IsValid {
		t.Errorf("tier=%q flags=%v status=%q valid=%v", res.Tier, res.Flags, res.Status, res.IsValid)
	}
	if res.SolvedCarry == nil || math.Abs(*res.SolvedCarry-0.02) > 0.02 {
		t.Errorf("solved carry %v", res.SolvedCarry)
	}
	if res.CallPutIVGap == nil || math.Abs(*res.CallPutIVGap) > 1e-6 {
		t.Errorf("iv gap %v", res.CallPutIVGap)
	}
	if res.Iterations < 1 || res.AccuratePricesUsed != 0 {
		t.Errorf("iterations=%d accurate=%d", res.Iterations, res.AccuratePricesUsed)
	}
	d := res.Diagnostics
	if d.TotalPairsReceived != 7 || d.MinSpread <= 0 || d.MaxSpread < d.MinSpread || d.PairsUsed < 2 {
		t.Errorf("diagnostics not populated: %+v", d)
	}
	if res.Forward <= 0 || res.PairsUsed != d.PairsUsed {
		t.Errorf("forward=%v pairsUsed=%d", res.Forward, res.PairsUsed)
	}
}

func TestImpliedForwardWithOptionsLevelsScheduleAndBudget(t *testing.T) {
	spot, df := 100.0, math.Exp(-0.01*0.5)
	quotes := blackQuotes(0.01)

	res, err := ImpliedForwardWithOptions(quotes, 0.5, ForwardAmericanEquity, &spot, &df,
		&ImpliedForwardOptions{Level: ForwardLevelAccurate})
	if err != nil {
		t.Fatalf("accurate: %v", err)
	}
	if res.Tier != "model_accurate" || res.AccuratePricesUsed != 2*len(quotes) {
		t.Errorf("accurate: tier=%q prices=%d", res.Tier, res.AccuratePricesUsed)
	}

	schedule := []ForwardDividend{{Time: 0.2, Amount: 0.5}}
	res, err = ImpliedForwardWithOptions(quotes, 0.5, ForwardAmericanEquity, &spot, &df,
		&ImpliedForwardOptions{Dividends: schedule, Level: ForwardLevelFast})
	if err != nil {
		t.Fatalf("fast with schedule: %v", err)
	}
	if res.Tier != "model_fast" || len(res.Flags) != 1 || res.Flags[0] != "schedule_ignored" {
		t.Errorf("fast with schedule: tier=%q flags=%v", res.Tier, res.Flags)
	}

	res, err = ImpliedForwardWithOptions(quotes, 0.5, ForwardAmericanEquity, &spot, &df,
		&ImpliedForwardOptions{Dividends: schedule})
	if err != nil {
		t.Fatalf("auto with schedule: %v", err)
	}
	if res.Tier != "model_accurate" || len(res.Flags) != 0 {
		t.Errorf("auto with schedule: tier=%q flags=%v", res.Tier, res.Flags)
	}

	budget := 1
	res, err = ImpliedForwardWithOptions(quotes, 0.5, ForwardAmericanEquity, &spot, &df,
		&ImpliedForwardOptions{Level: ForwardLevelAccurate, MaxAccuratePrices: &budget})
	if err != nil {
		t.Fatalf("budget: %v", err)
	}
	if res.Tier != "model_fast" || len(res.Flags) != 1 || res.Flags[0] != "latency_fallback" {
		t.Errorf("budget: tier=%q flags=%v", res.Tier, res.Flags)
	}
}

func TestImpliedForwardWithOptionsOtherTiersAndConventions(t *testing.T) {
	spot := 100.0
	res, err := ImpliedForwardWithOptions(blackQuotes(0.01), 0.5, ForwardAmericanEquity, &spot, nil, nil)
	if err != nil {
		t.Fatalf("corridor: %v", err)
	}
	if res.Tier != "corridor" || res.SolvedCarry != nil || res.CallPutIVGap != nil {
		t.Errorf("corridor: tier=%q carry=%v", res.Tier, res.SolvedCarry)
	}

	df := 1.0
	res, err = ImpliedForwardWithOptions(blackQuotes(0.5), 0.5, ForwardEuropeanVanilla, &spot, &df, nil)
	if err != nil {
		t.Fatalf("european: %v", err)
	}
	if res.Tier != "" || res.SolvedCarry != nil || len(res.Flags) != 0 {
		t.Errorf("european: tier=%q carry=%v flags=%v", res.Tier, res.SolvedCarry, res.Flags)
	}

	if _, err := ImpliedForwardWithOptions(blackQuotes(0.01), 0.5, ForwardAmericanEquity, nil, nil, nil); err == nil {
		t.Error("American convention without spot must fail")
	}
	if _, err := ImpliedForwardWithOptions(nil, 0.5, ForwardEuropeanVanilla, nil, nil, nil); err == nil {
		t.Error("empty quotes must fail")
	}
}

func TestImpliedForwardWithOptionsRejectsInvalidOptions(t *testing.T) {
	spot, df := 100.0, math.Exp(-0.01*0.5)
	quotes := blackQuotes(0.01)
	cases := map[string]*ImpliedForwardOptions{
		"level":          {Level: ForwardLevel(9)},
		"inverted range": {CarrySearchRange: &[2]float64{0.25, -0.10}},
		"late dividend":  {Dividends: []ForwardDividend{{Time: 0.9, Amount: 1}}},
	}
	for name, opts := range cases {
		if _, err := ImpliedForwardWithOptions(quotes, 0.5, ForwardAmericanEquity, &spot, &df, opts); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestImpliedForwardWithOptionsAppliesOptionalSettings(t *testing.T) {
	spot, df := 100.0, math.Exp(-0.01*0.5)
	vol, kappa, tol := 0.25, 1.2, 1e-8
	passes := 0
	opts := &ImpliedForwardOptions{
		AtmVol:           &vol,
		Level:            ForwardLevelAccurate,
		CorrectionPasses: &passes,
		CarryTolerance:   &tol,
		CorridorKappa:    &kappa,
		CarrySearchRange: &[2]float64{-0.05, 0.20},
		CarryFlagRange:   &[2]float64{-0.02, 0.10},
	}
	res, err := ImpliedForwardWithOptions(blackQuotes(0.01), 0.5, ForwardAmericanEquity, &spot, &df, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Zero correction passes means the fast level.
	if res.Tier != "model_fast" {
		t.Errorf("tier=%q", res.Tier)
	}
}
