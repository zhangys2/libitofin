package itofin

import (
	"math"
	"testing"
)

func TestTotalVarianceCubicSmileSection(t *testing.T) {
	session, err := NewSession()
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()

	const forward, expiry, atm = 100.0, 0.5, 0.20
	scale := atm * math.Sqrt(expiry)
	strike := func(x float64) float64 { return forward * math.Exp(x*scale) }
	points := []float64{-1.5, -1.0, -0.6, 0.0, 0.6, 1.0, 1.5}
	strikes := make([]float64, len(points))
	midIVs := make([]float64, len(points))
	for i, point := range points {
		strikes[i] = strike(point)
		midIVs[i] = 0.20 + 0.02*point*point
	}

	smile, err := session.TotalVarianceCubicSmileSection(strikes, midIVs, forward, expiry, atm)
	if err != nil {
		t.Fatalf("TotalVarianceCubicSmileSection failed: %v", err)
	}

	// ATM volatility
	volATM, err := smile.Volatility(forward)
	if err != nil {
		t.Fatalf("Volatility(ATM) failed: %v", err)
	}
	if math.Abs(volATM-0.20) > 0.01 {
		t.Fatalf("expected ATM vol close to 0.20, got %v", volATM)
	}

	// Total variance at log-moneyness 0
	tvATM, err := smile.TotalVariance(0.0)
	if err != nil {
		t.Fatalf("TotalVariance(0) failed: %v", err)
	}
	if math.Abs(tvATM-volATM*volATM*expiry) > 1e-6 {
		t.Fatalf("expected TV close to %v, got %v", volATM*volATM*expiry, tvATM)
	}

	// Roger Lee wing slopes
	rSlope, err := smile.RightWingSlope()
	if err != nil || rSlope < 0 || rSlope > 2.0 {
		t.Fatalf("invalid right wing slope %v (err: %v)", rSlope, err)
	}

	lSlope, err := smile.LeftWingSlope()
	if err != nil || lSlope > 0 || lSlope < -2.0 {
		t.Fatalf("invalid left wing slope %v (err: %v)", lSlope, err)
	}

	// Butterfly arbitrage check
	report, err := smile.ButterflyReport()
	if err != nil {
		t.Fatalf("ButterflyReport failed: %v", err)
	}
	if report.HasArbitrage {
		t.Fatalf("expected no arbitrage, got report: %+v", report)
	}
}
