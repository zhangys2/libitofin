package itofin

/*
#include "itofin.h"
*/
import "C"
import "unsafe"

// CubicSmileSection is one expiry's mid-IV smile on standardized log-moneyness.
type CubicSmileSection struct{ object }

// CubicSmileSection fits a natural cubic through paired strike and mid-IV observations.
// A nil stdDevPoints slice selects the default report grid.
func (s *Session) CubicSmileSection(strikes, midIVs []float64, forward, exerciseTime, atmVol float64, stdDevPoints []float64, extrapolate bool) (*CubicSmileSection, error) {
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
		return ffiError(C.itofin_cubic_smile_new(
			s.ctx,
			(*C.double)(unsafe.Pointer(&strikes[0])), C.size_t(len(strikes)),
			(*C.double)(unsafe.Pointer(&midIVs[0])), C.size_t(len(midIVs)),
			C.double(forward), C.double(exerciseTime), C.double(atmVol),
			points, C.size_t(len(stdDevPoints)), ex, &id, &e,
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

// MinStrike is the lowest source strike.
func (v *CubicSmileSection) MinStrike() (float64, error) { return v.query(6, 0) }

// MaxStrike is the highest source strike.
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

// StdDevPoints is the configured evaluation grid.
func (v *CubicSmileSection) StdDevPoints() ([]float64, error) {
	values, _, err := v.series(0)
	return values, err
}

// NodeStdDevPoints are the sorted source coordinates.
func (v *CubicSmileSection) NodeStdDevPoints() ([]float64, error) {
	values, _, err := v.series(1)
	return values, err
}

// NodeMidIVs are the source mid volatilities paired with NodeStdDevPoints.
func (v *CubicSmileSection) NodeMidIVs() ([]float64, error) {
	values, _, err := v.series(2)
	return values, err
}

// SampledMidIVs evaluates the report grid. The bool is false outside the observed domain.
func (v *CubicSmileSection) SampledMidIVs() ([]float64, []bool, error) {
	return v.series(3)
}

// NodeResiduals are fitted minus observed mid-IV at each source node.
func (v *CubicSmileSection) NodeResiduals() ([]float64, error) {
	values, _, err := v.series(4)
	return values, err
}

// SegmentCoefficients are the cubic (a, b, c) terms on each adjacent node pair.
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
