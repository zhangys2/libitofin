package itofin

/*
#include "itofin.h"
*/
import "C"
import (
	"fmt"
	"math"
	"unsafe"
)

// OptionQuotePair represents a two-sided option quote pair at a single strike.
type OptionQuotePair struct {
	Strike  float64
	CallBid float64
	CallAsk float64
	PutBid  float64
	PutAsk  float64
}

// NewOptionQuotePair constructs a new OptionQuotePair.
func NewOptionQuotePair(strike, callBid, callAsk, putBid, putAsk float64) *OptionQuotePair {
	return &OptionQuotePair{
		Strike:  strike,
		CallBid: callBid,
		CallAsk: callAsk,
		PutBid:  putBid,
		PutAsk:  putAsk,
	}
}

// IsValid checks whether both call and put quotes have non-negative bid, positive ask, and ask >= bid.
func (p *OptionQuotePair) IsValid() bool {
	return p.CallBid >= 0 && p.CallAsk > 0 && p.CallAsk >= p.CallBid && p.PutBid >= 0 && p.PutAsk > 0 && p.PutAsk >= p.PutBid
}

// Repr returns the string representation.
func (p *OptionQuotePair) Repr() string {
	return fmt.Sprintf("OptionQuotePair(strike=%v, call=[%v, %v], put=[%v, %v])", p.Strike, p.CallBid, p.CallAsk, p.PutBid, p.PutAsk)
}

// ForwardDiagnostics holds diagnostic health metrics for the calculated forward.
type ForwardDiagnostics struct {
	Status                 string
	IsValid                bool
	TotalPairsReceived     int
	PairsUsed              int
	PairsPruned            int
	IsCrossed              bool
	BoxArbitrageViolations int
	WlsRmse                float64
	MinSpread              float64
	MaxSpread              float64
}

// ImpliedForwardResult holds the result of the implied forward estimation.
type ImpliedForwardResult struct {
	Forward          float64
	ForwardBidStrict float64
	ForwardAskStrict float64
	ForwardBidRobust float64
	ForwardAskRobust float64
	DiscountFactor   float64
	ImpliedCarryRate *float64
	Diagnostics      ForwardDiagnostics
	IsValid          bool
	PairsUsed        int
	Status           string
}

// Repr returns the string representation.
func (r *ImpliedForwardResult) Repr() string {
	return fmt.Sprintf("ImpliedForwardResult(forward=%v, discount_factor=%v, status=%s)", r.Forward, r.DiscountFactor, r.Status)
}

// forwardStatus decodes the status_code of itofin_implied_forward_calculate:
// 0 Valid, 1 CrossedSyntheticQuotes, 2 BoxSpreadArbitrage, 3 ImplausibleDiscountFactor,
// 4 SpotDeviationExceeded, 5 DegradedAmericanBounds, 6 ImplausibleCarry. The name of code 0 stays "Ok" for
// existing callers; the others are the core's ForwardStatus names, as in the Python binding.
// Only code 0 is valid, as in Python's is_valid. A new Rust ForwardStatus variant needs a case here
// (and a code in forward_api.rs); until then it decodes as Unknown(code).
func forwardStatus(code int32) (name string, valid bool) {
	switch code {
	case 0:
		return "Ok", true
	case 1:
		return "WarningCrossedSyntheticQuotes", false
	case 2:
		return "WarningBoxSpreadArbitrage", false
	case 3:
		return "WarningImplausibleDiscountFactor", false
	case 4:
		return "WarningSpotDeviationExceeded", false
	case 5:
		return "DegradedAmericanBounds", false
	case 6:
		return "WarningImplausibleCarry", false
	default:
		return fmt.Sprintf("Unknown(%d)", code), false
	}
}

// forwardCarryRate returns nil when the core reports no carry rate. The FFI encodes None as NaN;
// an infinite value is a computed rate and is passed through, as in the Python binding.
func forwardCarryRate(v float64) *float64 {
	if math.IsNaN(v) {
		return nil
	}
	return &v
}

// ImpliedForward calculates the implied forward price and discount factor from option quote pairs.
// convention: 0 for EuropeanVanilla, 1 for CryptoInverse, 2 for AmericanEquity (which requires spot).
// MinSpread, MaxSpread and BoxArbitrageViolations are not reported by this entry point and stay 0.
func ImpliedForward(pairs []OptionQuotePair, expiryYears float64, convention int32, spot *float64, discountFactor *float64) (*ImpliedForwardResult, error) {
	if len(pairs) == 0 {
		return nil, fmt.Errorf("quotes slice cannot be empty")
	}
	count := len(pairs)
	strikes := make([]float64, count)
	callBids := make([]float64, count)
	callAsks := make([]float64, count)
	putBids := make([]float64, count)
	putAsks := make([]float64, count)

	for i, p := range pairs {
		strikes[i] = p.Strike
		callBids[i] = p.CallBid
		callAsks[i] = p.CallAsk
		putBids[i] = p.PutBid
		putAsks[i] = p.PutAsk
	}

	var spotPtr *C.double
	var spotVal C.double
	if spot != nil {
		spotVal = C.double(*spot)
		spotPtr = &spotVal
	}

	var dfPtr *C.double
	var dfVal C.double
	if discountFactor != nil {
		dfVal = C.double(*discountFactor)
		dfPtr = &dfVal
	}

	var out C.ItofinImpliedForwardResult
	var e C.ItofinError

	err := ffiError(C.itofin_implied_forward_calculate(
		(*C.double)(unsafe.Pointer(&strikes[0])),
		(*C.double)(unsafe.Pointer(&callBids[0])),
		(*C.double)(unsafe.Pointer(&callAsks[0])),
		(*C.double)(unsafe.Pointer(&putBids[0])),
		(*C.double)(unsafe.Pointer(&putAsks[0])),
		C.size_t(count),
		C.int32_t(convention),
		spotPtr,
		dfPtr,
		C.double(expiryYears),
		&out,
		&e,
	), &e)
	if err != nil {
		return nil, err
	}

	statusStr, isValid := forwardStatus(int32(out.status_code))
	// The core sets is_crossed from the strict bounds whatever the status: a warning raised earlier
	// (e.g. an implausible discount factor) hides the crossed status but not the crossed book.
	isCrossed := float64(out.forward_bid_strict) > float64(out.forward_ask_strict)
	carryRate := forwardCarryRate(float64(out.implied_carry_rate))

	diag := ForwardDiagnostics{
		Status:                 statusStr,
		IsValid:                isValid,
		TotalPairsReceived:     count,
		PairsUsed:              int(out.pairs_used),
		PairsPruned:            int(out.pairs_pruned),
		IsCrossed:              isCrossed,
		BoxArbitrageViolations: 0,
		WlsRmse:                float64(out.wls_rmse),
		MinSpread:              0,
		MaxSpread:              0,
	}

	return &ImpliedForwardResult{
		Forward:          float64(out.forward),
		ForwardBidStrict: float64(out.forward_bid_strict),
		ForwardAskStrict: float64(out.forward_ask_strict),
		ForwardBidRobust: float64(out.forward_bid_robust),
		ForwardAskRobust: float64(out.forward_ask_robust),
		DiscountFactor:   float64(out.discount_factor),
		ImpliedCarryRate: carryRate,
		Diagnostics:      diag,
		IsValid:          isValid,
		PairsUsed:        int(out.pairs_used),
		Status:           statusStr,
	}, nil
}

// ForwardConvention selects how option quotes are interpreted.
type ForwardConvention int32

// Convention codes of the C ABI.
const (
	// ForwardEuropeanVanilla uses cash or physically settled put-call parity.
	ForwardEuropeanVanilla ForwardConvention = 0
	// ForwardCryptoInverse uses coin-margined parity with premiums quoted in the coin.
	ForwardCryptoInverse ForwardConvention = 1
	// ForwardAmericanEquity de-Americanizes American quotes; it requires spot.
	ForwardAmericanEquity ForwardConvention = 2
)

// ForwardLevel is the accuracy level of the model-based American path.
type ForwardLevel int32

// Level codes of the C ABI.
const (
	// ForwardLevelAuto uses the accurate level when a dividend schedule is given.
	ForwardLevelAuto ForwardLevel = 0
	// ForwardLevelFast uses the closed-form pricer only and ignores (and flags) a schedule.
	ForwardLevelFast ForwardLevel = 1
	// ForwardLevelAccurate adds premium-correction passes with a tree or grid pricer.
	ForwardLevelAccurate ForwardLevel = 2
)

// ForwardDividend is a discrete cash dividend, with Time in years on the basis of expiryYears.
type ForwardDividend struct {
	Time   float64
	Amount float64
}

// ImpliedForwardOptions are the settings of the model-based American path. A nil pointer or a
// zero field keeps the default.
type ImpliedForwardOptions struct {
	// AtmVol sizes the strike window; estimated from the quotes when nil and a discount factor is
	// given, otherwise 0.30.
	AtmVol *float64
	// Dividends is the discrete cash-dividend schedule, each time in (0, expiryYears].
	Dividends []ForwardDividend
	// Level is the accuracy level; the zero value is ForwardLevelAuto.
	Level ForwardLevel
	// CarrySearchRange is the interval searched for the solved carry per year (default -0.10, 0.25).
	CarrySearchRange *[2]float64
	// CarryFlagRange is the interval outside of which the status is WarningImplausibleCarry
	// (default -0.05, 0.15); it must lie inside the search range.
	CarryFlagRange *[2]float64
	// CorrectionPasses overrides the premium-correction passes; 0 means the fast level.
	CorrectionPasses *int
	// CarryTolerance is the absolute tolerance on the solved carry (default 1e-7).
	CarryTolerance *float64
	// CorridorKappa sizes the model path's strike window in standard deviations (default 1.0).
	CorridorKappa *float64
	// MaxAccuratePrices is the work budget of accurate-pricer evaluations (default 64).
	MaxAccuratePrices *int
}

// ImpliedForwardResultEx is an ImpliedForwardResult with the American model-path outputs.
type ImpliedForwardResultEx struct {
	ImpliedForwardResult
	// Tier is "corridor", "model_fast" or "model_accurate" for the American convention, "" otherwise.
	Tier string
	// Flags lists "schedule_ignored", "latency_fallback" and "non_converged" when set.
	Flags []string
	// SolvedCarry is the solved continuous yield (or the residual carry beside a schedule);
	// nil unless a model tier produced the result.
	SolvedCarry *float64
	// CallPutIVGap is the mean call minus put implied volatility at the solution (model tiers only).
	CallPutIVGap *float64
	// Iterations is the number of gap evaluations of the carry solve (model tiers only).
	Iterations int
	// AccuratePricesUsed is the number of accurate-pricer evaluations spent (model tiers only).
	AccuratePricesUsed int
}

// forwardTier names the tier_code of itofin_implied_forward_calculate_with_options.
func forwardTier(code int32) string {
	switch code {
	case -1:
		return ""
	case 0:
		return "corridor"
	case 1:
		return "model_fast"
	case 2:
		return "model_accurate"
	default:
		return fmt.Sprintf("unknown(%d)", code)
	}
}

// forwardFlags names the set bits of the flags field; bits this version does not know are
// reported as one "unknown(0x..)" entry.
func forwardFlags(bits uint32) []string {
	var names []string
	for _, f := range []struct {
		bit  uint32
		name string
	}{{1, "schedule_ignored"}, {2, "latency_fallback"}, {4, "non_converged"}} {
		if bits&f.bit != 0 {
			names = append(names, f.name)
			bits &^= f.bit
		}
	}
	if bits != 0 {
		names = append(names, fmt.Sprintf("unknown(0x%x)", bits))
	}
	return names
}

func nativeFlag(set bool) C.int32_t {
	if set {
		return 1
	}
	return 0
}

// ImpliedForwardWithOptions is ImpliedForward with the settings and outputs of the model-based
// American path. With the American convention and a discount factor the quotes are
// de-Americanized before the parity estimate; otherwise, or when the model cannot be solved,
// the model-free corridor estimate is returned with Tier "corridor". A nil opts keeps the defaults.
func ImpliedForwardWithOptions(pairs []OptionQuotePair, expiryYears float64, convention ForwardConvention, spot *float64, discountFactor *float64, opts *ImpliedForwardOptions) (*ImpliedForwardResultEx, error) {
	if len(pairs) == 0 {
		return nil, fmt.Errorf("quotes slice cannot be empty")
	}
	count := len(pairs)
	strikes := make([]float64, count)
	callBids := make([]float64, count)
	callAsks := make([]float64, count)
	putBids := make([]float64, count)
	putAsks := make([]float64, count)
	for i, p := range pairs {
		strikes[i] = p.Strike
		callBids[i] = p.CallBid
		callAsks[i] = p.CallAsk
		putBids[i] = p.PutBid
		putAsks[i] = p.PutAsk
	}

	var spotPtr *C.double
	var spotVal C.double
	if spot != nil {
		spotVal = C.double(*spot)
		spotPtr = &spotVal
	}
	var dfPtr *C.double
	var dfVal C.double
	if discountFactor != nil {
		dfVal = C.double(*discountFactor)
		dfPtr = &dfVal
	}

	var native C.ItofinImpliedForwardOptions
	var e C.ItofinError
	if err := ffiError(C.itofin_implied_forward_options_default(&native, &e), &e); err != nil {
		return nil, err
	}
	var pins runtimePinner
	defer pins.unpin()
	if opts != nil {
		if opts.AtmVol != nil {
			native.atm_vol = C.double(*opts.AtmVol)
			native.has_atm_vol = 1
		}
		if len(opts.Dividends) > 0 {
			times := make([]float64, len(opts.Dividends))
			amounts := make([]float64, len(opts.Dividends))
			for i, d := range opts.Dividends {
				times[i] = d.Time
				amounts[i] = d.Amount
			}
			pins.pin(times, amounts)
			native.dividend_times = (*C.double)(unsafe.Pointer(&times[0]))
			native.dividend_amounts = (*C.double)(unsafe.Pointer(&amounts[0]))
			native.dividend_count = C.size_t(len(times))
		}
		native.level = C.int32_t(opts.Level)
		if opts.CarrySearchRange != nil {
			native.carry_search_low = C.double(opts.CarrySearchRange[0])
			native.carry_search_high = C.double(opts.CarrySearchRange[1])
		}
		if opts.CarryFlagRange != nil {
			native.carry_flag_low = C.double(opts.CarryFlagRange[0])
			native.carry_flag_high = C.double(opts.CarryFlagRange[1])
		}
		if opts.CorrectionPasses != nil {
			if *opts.CorrectionPasses < 0 {
				return nil, fmt.Errorf("%w: CorrectionPasses must not be negative", ErrInvalidArgument)
			}
			native.correction_passes = C.size_t(*opts.CorrectionPasses)
			native.has_correction_passes = nativeFlag(true)
		}
		if opts.CarryTolerance != nil {
			native.carry_tolerance = C.double(*opts.CarryTolerance)
		}
		if opts.CorridorKappa != nil {
			native.corridor_kappa = C.double(*opts.CorridorKappa)
		}
		if opts.MaxAccuratePrices != nil {
			if *opts.MaxAccuratePrices < 0 {
				return nil, fmt.Errorf("%w: MaxAccuratePrices must not be negative", ErrInvalidArgument)
			}
			native.max_accurate_prices = C.size_t(*opts.MaxAccuratePrices)
		}
	}

	var out C.ItofinImpliedForwardResultEx
	err := ffiError(C.itofin_implied_forward_calculate_with_options(
		(*C.double)(unsafe.Pointer(&strikes[0])),
		(*C.double)(unsafe.Pointer(&callBids[0])),
		(*C.double)(unsafe.Pointer(&callAsks[0])),
		(*C.double)(unsafe.Pointer(&putBids[0])),
		(*C.double)(unsafe.Pointer(&putAsks[0])),
		C.size_t(count),
		C.int32_t(convention),
		spotPtr,
		dfPtr,
		C.double(expiryYears),
		&native,
		&out,
		&e,
	), &e)
	if err != nil {
		return nil, err
	}

	statusStr, isValid := forwardStatus(int32(out.core.status_code))
	diag := ForwardDiagnostics{
		Status:                 statusStr,
		IsValid:                isValid,
		TotalPairsReceived:     int(out.total_pairs_received),
		PairsUsed:              int(out.core.pairs_used),
		PairsPruned:            int(out.core.pairs_pruned),
		IsCrossed:              out.is_crossed != 0,
		BoxArbitrageViolations: int(out.box_arbitrage_violations),
		WlsRmse:                float64(out.core.wls_rmse),
		MinSpread:              float64(out.min_spread),
		MaxSpread:              float64(out.max_spread),
	}
	res := &ImpliedForwardResultEx{
		ImpliedForwardResult: ImpliedForwardResult{
			Forward:          float64(out.core.forward),
			ForwardBidStrict: float64(out.core.forward_bid_strict),
			ForwardAskStrict: float64(out.core.forward_ask_strict),
			ForwardBidRobust: float64(out.core.forward_bid_robust),
			ForwardAskRobust: float64(out.core.forward_ask_robust),
			DiscountFactor:   float64(out.core.discount_factor),
			ImpliedCarryRate: forwardCarryRate(float64(out.core.implied_carry_rate)),
			Diagnostics:      diag,
			IsValid:          isValid,
			PairsUsed:        int(out.core.pairs_used),
			Status:           statusStr,
		},
		Tier:  forwardTier(int32(out.tier_code)),
		Flags: forwardFlags(uint32(out.flags)),
	}
	if out.has_model != 0 {
		carry := float64(out.solved_carry)
		gap := float64(out.call_put_iv_gap)
		res.SolvedCarry = &carry
		res.CallPutIVGap = &gap
		res.Iterations = int(out.iterations)
		res.AccuratePricesUsed = int(out.accurate_prices_used)
	}
	return res, nil
}
