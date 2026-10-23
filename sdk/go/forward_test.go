package itofin

import (
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
