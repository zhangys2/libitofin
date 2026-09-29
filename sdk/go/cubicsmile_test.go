package itofin

import "math"
import "testing"

func TestCubicSmileMidIV(t *testing.T) {
	session, err := NewSession()
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	const forward, expiry, atm = 100.0, 1.0, 0.2
	scale := atm * math.Sqrt(expiry)
	strike := func(x float64) float64 { return forward * math.Exp(x*scale) }
	smile, err := session.CubicSmileSection(
		[]float64{strike(-1), strike(0), strike(1)},
		[]float64{0.28, 0.30, 0.32},
		forward, expiry, atm, nil, false,
	)
	if err != nil {
		t.Fatal(err)
	}
	got, err := smile.VolatilityAtStdDev(0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-0.30) > 1e-12 {
		t.Fatalf("ATM mid IV = %v", got)
	}
	fwd, err := smile.Forward()
	if err != nil || math.Abs(fwd-forward) > 1e-12 {
		t.Fatalf("forward = %v (%v)", fwd, err)
	}
}
