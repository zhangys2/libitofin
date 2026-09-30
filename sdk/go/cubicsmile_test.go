package itofin

import (
	"math"
	"testing"
)

func TestCubicSmileMidIV(t *testing.T) {
	session, err := NewSession()
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	const forward, expiry, atm = 100.0, 1.0, 0.2
	scale := atm * math.Sqrt(expiry)
	strike := func(x float64) float64 { return forward * math.Exp(x*scale) }
	points := []float64{-3, -1.5, -1, -0.6, 0, 0.6, 1, 1.5, 3}
	strikes := make([]float64, len(points))
	midIVs := make([]float64, len(points))
	for i, point := range points {
		strikes[i] = strike(point)
		midIVs[i] = 0.30 + 0.02*point
	}
	smile, err := session.CubicSmileSection(
		strikes, midIVs, forward, expiry, atm, nil, false,
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
	knots, err := smile.NodeStdDevPoints()
	if err != nil || len(knots) != len(points) {
		t.Fatalf("knot points = %v (%v)", knots, err)
	}
}

func TestCubicSmileSparseSmoothing(t *testing.T) {
	session, err := NewSession()
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	points := []float64{-1.4, -1, -0.6, 0, 0.6, 1, 1.4}
	strikes, vols := make([]float64, len(points)), make([]float64, len(points))
	for i, x := range points {
		strikes[i] = 100 * math.Exp(x*0.2)
		vols[i] = 0.3 + 0.02*x*x
	}
	smile, err := session.CubicSmileSection(strikes, vols, 100, 1, 0.2, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	knots, err := smile.NodeStdDevPoints()
	if err != nil || len(knots) != 9 {
		t.Fatalf("sparse fit knots = %v (%v)", knots, err)
	}
	weight, err := smile.Smoothing()
	if err != nil || weight != 0.01 {
		t.Fatalf("default smoothing = %v (%v)", weight, err)
	}
	if _, err := session.CubicSmileSectionWithSmoothing(strikes, vols, 100, 1, 0.2, nil, false, 0); err == nil {
		t.Fatal("unregularized seven-quote fit must fail")
	}
	custom, err := session.CubicSmileSectionWithSmoothing(strikes, vols, 100, 1, 0.2, nil, false, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	weight, err = custom.Smoothing()
	if err != nil || weight != 0.05 {
		t.Fatalf("custom smoothing = %v (%v)", weight, err)
	}
}

func TestCubicSmileObservationResiduals(t *testing.T) {
	session, err := NewSession()
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close()
	points := []float64{-2, -1, -0.5, 0, 0.5, 1, 2}
	vols := []float64{1.0, 0.28, 0.31, 0.29, 0.34, 0.33, 2.0}
	strikes := make([]float64, len(points))
	for i, x := range points {
		strikes[i] = 100 * math.Exp(x*0.2)
	}
	smile, err := session.CubicSmileSection(strikes, vols, 100, 1, 0.2, []float64{-1, 1}, false)
	if err != nil {
		t.Fatal(err)
	}
	observedStrikes, err := smile.ObservedStrikes()
	if err != nil || len(observedStrikes) != len(points) {
		t.Fatalf("observed strikes = %v (%v)", observedStrikes, err)
	}
	observedX, err := smile.ObservedStdDevPoints()
	if err != nil || len(observedX) != len(points) {
		t.Fatalf("observed coordinates = %v (%v)", observedX, err)
	}
	observedIVs, err := smile.ObservedMidIVs()
	if err != nil || len(observedIVs) != len(points) {
		t.Fatalf("observed IVs = %v (%v)", observedIVs, err)
	}
	residuals, valid, err := smile.ObservationResiduals()
	if err != nil || len(residuals) != len(points) || len(valid) != len(points) {
		t.Fatalf("residuals = %v, valid = %v (%v)", residuals, valid, err)
	}
	for i := range points {
		if math.Abs(observedStrikes[i]-strikes[i]) > 1e-10 ||
			math.Abs(observedX[i]-points[i]) > 1e-12 ||
			math.Abs(observedIVs[i]-vols[i]) > 1e-12 {
			t.Fatalf("observation %d was not preserved", i)
		}
		if valid[i] != (i > 0 && i < len(points)-1) {
			t.Fatalf("unexpected fit flag at %d", i)
		}
		if valid[i] {
			fitted, err := smile.VolatilityAtStdDev(points[i])
			if err != nil || math.Abs(residuals[i]-(fitted-vols[i])) > 1e-12 {
				t.Fatalf("residual at %d = %v (%v)", i, residuals[i], err)
			}
		}
	}
}
