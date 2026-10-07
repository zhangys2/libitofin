package itofin

/*
#include "itofin.h"
*/
import "C"
import (
	"fmt"
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

// ImpliedForward calculates the implied forward price and discount factor from option quote pairs.
// convention: 0 for EuropeanVanilla, 1 for American, 2 for CryptoInverse.
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

	statusStr := "Ok"
	isValid := true
	if out.status_code != 0 {
		isValid = false
		switch out.status_code {
		case 1:
			statusStr = "InsufficientPairs"
		case 2:
			statusStr = "ArbitrageViolations"
		case 3:
			statusStr = "CrossedMarket"
		default:
			statusStr = "FailedFit"
		}
	}

	var carryRate *float64
	if expiryYears > 0 {
		r := float64(out.implied_carry_rate)
		carryRate = &r
	}

	diag := ForwardDiagnostics{
		Status:                 statusStr,
		IsValid:                isValid,
		TotalPairsReceived:     count,
		PairsUsed:              int(out.pairs_used),
		PairsPruned:            int(out.pairs_pruned),
		IsCrossed:              out.status_code == 3,
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
