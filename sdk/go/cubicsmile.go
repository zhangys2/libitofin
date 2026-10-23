package itofin

/*
#include "itofin.h"
*/
import "C"
import "unsafe"

// CubicSmileSection is one expiry's mid-IV smile on standardized log-moneyness.
type CubicSmileSection struct{ object }

// CubicSmileSection fits fixed knots using mean squared IV error plus a 0.01
// integrated-curvature penalty. A nil stdDevPoints slice selects the default nine knots.
func (s *Session) CubicSmileSection(strikes, midIVs []float64, forward, exerciseTime, atmVol float64, stdDevPoints []float64, extrapolate bool) (*CubicSmileSection, error) {
	return s.CubicSmileSectionWithSmoothing(strikes, midIVs, forward, exerciseTime, atmVol, stdDevPoints, extrapolate, 0.01)
}

// CubicSmileSectionWithSmoothing sets an explicit nonnegative curvature penalty.
// Positive smoothing permits two distinct in-range quotes; zero requires full quote rank.
func (s *Session) CubicSmileSectionWithSmoothing(strikes, midIVs []float64, forward, exerciseTime, atmVol float64, stdDevPoints []float64, extrapolate bool, smoothing float64) (*CubicSmileSection, error) {
	if len(strikes) == 0 || len(midIVs) == 0 {
		return nil, errNilArgument("strikes and mid IVs")
	}
	var id C.uint64_t
	ex := C.uint8_t(0)
	if extrapolate {
		ex = 1
	}
	err := s.invoke(func() error {
		var e C.ItofinError
		var points *C.double
		if len(stdDevPoints) > 0 {
			points = (*C.double)(unsafe.Pointer(&stdDevPoints[0]))
		}
		return ffiError(C.itofin_cubic_smile_new_with_smoothing(
			s.ctx,
			(*C.double)(unsafe.Pointer(&strikes[0])), C.size_t(len(strikes)),
			(*C.double)(unsafe.Pointer(&midIVs[0])), C.size_t(len(midIVs)),
			C.double(forward), C.double(exerciseTime), C.double(atmVol),
			points, C.size_t(len(stdDevPoints)), ex, C.double(smoothing), &id, &e,
		), &e)
	})
	if err != nil {
		return nil, err
	}
	return &CubicSmileSection{object{s, uint64(id)}}, nil
}

func (v *CubicSmileSection) query(kind int32, x float64) (float64, error) {
	var out C.double
	err := v.session.invoke(func() error {
		var e C.ItofinError
		return ffiError(C.itofin_cubic_smile_query(v.session.ctx, C.uint64_t(v.id), C.int32_t(kind), C.double(x), &out, &e), &e)
	})
	return float64(out), err
}

// Volatility is the fitted mid IV at a strike.
func (v *CubicSmileSection) Volatility(strike float64) (float64, error) {
	return v.query(0, strike)
}

// Variance is the Black variance at a strike.
func (v *CubicSmileSection) Variance(strike float64) (float64, error) {
	return v.query(8, strike)
}

// VolatilityAtStdDev is the fitted mid IV at a signed standard-deviation point.
func (v *CubicSmileSection) VolatilityAtStdDev(point float64) (float64, error) {
	return v.query(1, point)
}

// StrikeAtStdDev maps a signed standard-deviation point back to a strike.
func (v *CubicSmileSection) StrikeAtStdDev(point float64) (float64, error) {
	return v.query(2, point)
}

// Forward is the ATM level used to standardize strikes.
func (v *CubicSmileSection) Forward() (float64, error) { return v.query(3, 0) }

// AtmVol is the fixed volatility that scales log-moneyness.
func (v *CubicSmileSection) AtmVol() (float64, error) { return v.query(4, 0) }

// ExerciseTime is the expiry year fraction.
func (v *CubicSmileSection) ExerciseTime() (float64, error) { return v.query(5, 0) }

// Smoothing is the integrated squared-curvature penalty weight.
func (v *CubicSmileSection) Smoothing() (float64, error) { return v.query(9, 0) }

// MinStrike is the strike at the lower knot-domain boundary.
func (v *CubicSmileSection) MinStrike() (float64, error) { return v.query(6, 0) }

// MaxStrike is the strike at the upper knot-domain boundary.
func (v *CubicSmileSection) MaxStrike() (float64, error) { return v.query(7, 0) }

func (v *CubicSmileSection) series(kind int32) ([]float64, []bool, error) {
	var length C.size_t
	err := v.session.invoke(func() error {
		var e C.ItofinError
		return ffiError(C.itofin_cubic_smile_series(v.session.ctx, C.uint64_t(v.id), C.int32_t(kind), nil, nil, 0, &length, &e), &e)
	})
	if err != nil {
		return nil, nil, err
	}
	n := int(length)
	values := make([]float64, n)
	flags := make([]byte, n)
	if n == 0 {
		return values, nil, nil
	}
	err = v.session.invoke(func() error {
		var e C.ItofinError
		return ffiError(C.itofin_cubic_smile_series(
			v.session.ctx, C.uint64_t(v.id), C.int32_t(kind),
			(*C.double)(unsafe.Pointer(&values[0])),
			(*C.uint8_t)(unsafe.Pointer(&flags[0])),
			C.size_t(n), &length, &e,
		), &e)
	})
	if err != nil {
		return nil, nil, err
	}
	present := make([]bool, n)
	for i, flag := range flags {
		present[i] = flag != 0
	}
	return values, present, nil
}

// StdDevPoints are the configured spline knots.
func (v *CubicSmileSection) StdDevPoints() ([]float64, error) {
	values, _, err := v.series(0)
	return values, err
}

// NodeStdDevPoints are the fixed spline-knot coordinates.
func (v *CubicSmileSection) NodeStdDevPoints() ([]float64, error) {
	values, _, err := v.series(1)
	return values, err
}

// NodeMidIVs are fitted knot IVs paired with NodeStdDevPoints.
func (v *CubicSmileSection) NodeMidIVs() ([]float64, error) {
	values, _, err := v.series(2)
	return values, err
}

// SampledMidIVs returns fitted IVs at knots. The bool is false outside the knot domain.
func (v *CubicSmileSection) SampledMidIVs() ([]float64, []bool, error) {
	return v.series(3)
}

// NodeResiduals are fitted minus fitted knot IVs, expected to be zero up to rounding.
func (v *CubicSmileSection) NodeResiduals() ([]float64, error) {
	values, _, err := v.series(4)
	return values, err
}

// ObservedStrikes are sorted source market strikes, including out-of-range quotes.
func (v *CubicSmileSection) ObservedStrikes() ([]float64, error) {
	values, _, err := v.series(6)
	return values, err
}

// ObservedStdDevPoints are standardized coordinates of source market quotes.
func (v *CubicSmileSection) ObservedStdDevPoints() ([]float64, error) {
	values, _, err := v.series(7)
	return values, err
}

// ObservedMidIVs are source IVs paired with ObservedStrikes.
func (v *CubicSmileSection) ObservedMidIVs() ([]float64, error) {
	values, _, err := v.series(8)
	return values, err
}

// ObservationResiduals returns fitted-minus-observed IV and validity flags.
// A false flag marks a quote outside the knot range, excluded from the fit.
func (v *CubicSmileSection) ObservationResiduals() ([]float64, []bool, error) {
	return v.series(9)
}

// SegmentCoefficients are the cubic (a, b, c) terms on each adjacent knot pair.
func (v *CubicSmileSection) SegmentCoefficients() ([][3]float64, error) {
	flat, _, err := v.series(5)
	if err != nil {
		return nil, err
	}
	coeffs := make([][3]float64, len(flat)/3)
	for i := range coeffs {
		coeffs[i] = [3]float64{flat[3*i], flat[3*i+1], flat[3*i+2]}
	}
	return coeffs, nil
}
