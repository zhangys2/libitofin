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
// 4 SpotDeviationExceeded, 5 DegradedAmericanBounds. The name of code 0 stays "Ok" for
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
