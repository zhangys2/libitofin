package itofin

/*
#include "itofin.h"
*/
import "C"
import "unsafe"

// ButterflyArbitrageReport holds diagnostics on risk-neutral density non-negativity.
type ButterflyArbitrageReport struct {
	MinDensity     float64
	ArgminK        float64
	HasArbitrage   bool
	Tolerance      float64
	PointsChecked  int
	FinalSmoothing float64
	RampIterations int
}

// TotalVarianceCubicSmileSection is a one-expiry cubic smile in total variance space with Roger Lee wing asymptotics.
type TotalVarianceCubicSmileSection struct{ object }

// TotalVarianceCubicSmileSection fits a cubic smile in total variance coordinates with Roger Lee wing asymptotics and butterfly arbitrage repair.
func (s *Session) TotalVarianceCubicSmileSection(strikes, midIVs []float64, forward, exerciseTime, atmVol float64) (*TotalVarianceCubicSmileSection, error) {
	return s.TotalVarianceCubicSmileSectionWithOptions(strikes, midIVs, forward, exerciseTime, atmVol, 0.01, true)
}

// TotalVarianceCubicSmileSectionWithOptions fits with explicit smoothing and arbitrage repair flag.
func (s *Session) TotalVarianceCubicSmileSectionWithOptions(strikes, midIVs []float64, forward, exerciseTime, atmVol, smoothing float64, arbitrageRepair bool) (*TotalVarianceCubicSmileSection, error) {
	if len(strikes) == 0 || len(midIVs) == 0 {
		return nil, errNilArgument("strikes and mid IVs")
	}
	var id C.uint64_t
	repair := C.uint8_t(0)
	if arbitrageRepair {
		repair = 1
	}
	err := s.invoke(func() error {
		var e C.ItofinError
		return ffiError(C.itofin_total_variance_cubic_smile_new(
			s.ctx,
			(*C.double)(unsafe.Pointer(&strikes[0])), C.size_t(len(strikes)),
			(*C.double)(unsafe.Pointer(&midIVs[0])), C.size_t(len(midIVs)),
			C.double(forward), C.double(exerciseTime), C.double(atmVol),
			C.double(smoothing), repair, &id, &e,
		), &e)
	})
	if err != nil {
		return nil, err
	}
	return &TotalVarianceCubicSmileSection{object{s, uint64(id)}}, nil
}

func (v *TotalVarianceCubicSmileSection) query(kind int32, x float64) (float64, error) {
	var out C.double
	err := v.session.invoke(func() error {
		var e C.ItofinError
		return ffiError(C.itofin_total_variance_cubic_smile_query(v.session.ctx, C.uint64_t(v.id), C.int32_t(kind), C.double(x), &out, &e), &e)
	})
	return float64(out), err
}

// Volatility returns the fitted Black implied volatility at strike.
func (v *TotalVarianceCubicSmileSection) Volatility(strike float64) (float64, error) {
	return v.query(0, strike)
}

// Variance returns total variance sigma^2 * T at strike.
func (v *TotalVarianceCubicSmileSection) Variance(strike float64) (float64, error) {
	return v.query(1, strike)
}

// TotalVariance returns total variance w(k) at log-moneyness k = ln(K/F).
func (v *TotalVarianceCubicSmileSection) TotalVariance(logMoneyness float64) (float64, error) {
	return v.query(2, logMoneyness)
}

// TotalVarianceDerivative returns first derivative w'(k) = dw/dk.
func (v *TotalVarianceCubicSmileSection) TotalVarianceDerivative(logMoneyness float64) (float64, error) {
	return v.query(3, logMoneyness)
}

// TotalVarianceSecondDerivative returns second derivative w”(k) = d^2w/dk^2.
func (v *TotalVarianceCubicSmileSection) TotalVarianceSecondDerivative(logMoneyness float64) (float64, error) {
	return v.query(4, logMoneyness)
}

// DurrlemanDensity returns the risk-neutral density g(k).
func (v *TotalVarianceCubicSmileSection) DurrlemanDensity(logMoneyness float64) (float64, error) {
	return v.query(5, logMoneyness)
}

// Forward returns the underlying forward price.
func (v *TotalVarianceCubicSmileSection) Forward() (float64, error) {
	return v.query(6, 0)
}

// ATMVol returns the at-the-money volatility.
func (v *TotalVarianceCubicSmileSection) ATMVol() (float64, error) {
	return v.query(7, 0)
}

// ExerciseTime returns expiry time in years.
func (v *TotalVarianceCubicSmileSection) ExerciseTime() (float64, error) {
	return v.query(8, 0)
}

// Smoothing returns the curvature penalty parameter.
func (v *TotalVarianceCubicSmileSection) Smoothing() (float64, error) {
	return v.query(9, 0)
}

// MinStrike returns the lower boundary strike.
func (v *TotalVarianceCubicSmileSection) MinStrike() (float64, error) {
	return v.query(10, 0)
}

// MaxStrike returns the upper boundary strike.
func (v *TotalVarianceCubicSmileSection) MaxStrike() (float64, error) {
	return v.query(11, 0)
}

// RightWingSlope returns asymptotic slope beta_R.
func (v *TotalVarianceCubicSmileSection) RightWingSlope() (float64, error) {
	return v.query(12, 0)
}

// LeftWingSlope returns asymptotic slope beta_L.
func (v *TotalVarianceCubicSmileSection) LeftWingSlope() (float64, error) {
	return v.query(13, 0)
}

// KnotsK returns the knot points in log-moneyness coordinates.
func (v *TotalVarianceCubicSmileSection) KnotsK() ([]float64, error) {
	return v.series(0)
}

// FittedTotalVariances returns the fitted total variances at knots.
func (v *TotalVarianceCubicSmileSection) FittedTotalVariances() ([]float64, error) {
	return v.series(1)
}

func (v *TotalVarianceCubicSmileSection) series(kind int32) ([]float64, error) {
	var length C.size_t
	err := v.session.invoke(func() error {
		var e C.ItofinError
		return ffiError(C.itofin_total_variance_cubic_smile_series(
			v.session.ctx, C.uint64_t(v.id), C.int32_t(kind), nil, 0, &length, &e,
		), &e)
	})
	if err != nil {
		return nil, err
	}
	out := make([]float64, int(length))
	if len(out) == 0 {
		return out, nil
	}
	err = v.session.invoke(func() error {
		var e C.ItofinError
		return ffiError(C.itofin_total_variance_cubic_smile_series(
			v.session.ctx, C.uint64_t(v.id), C.int32_t(kind), (*C.double)(unsafe.Pointer(&out[0])), length, &length, &e,
		), &e)
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ButterflyReport checks for butterfly arbitrage across the smile.
func (v *TotalVarianceCubicSmileSection) ButterflyReport() (*ButterflyArbitrageReport, error) {
	var minD, argK, finalSmoothing, tol C.double
	var hasArb C.uint8_t
	var rampIterations C.uint32_t
	var pointsChecked C.size_t
	err := v.session.invoke(func() error {
		var e C.ItofinError
		return ffiError(C.itofin_total_variance_cubic_smile_check_arbitrage(
			v.session.ctx, C.uint64_t(v.id), &minD, &argK, &hasArb, &finalSmoothing, &rampIterations, &tol, &pointsChecked, &e,
		), &e)
	})
	if err != nil {
		return nil, err
	}
	return &ButterflyArbitrageReport{
		MinDensity:     float64(minD),
		ArgminK:        float64(argK),
		HasArbitrage:   hasArb != 0,
		Tolerance:      float64(tol),
		PointsChecked:  int(pointsChecked),
		FinalSmoothing: float64(finalSmoothing),
		RampIterations: int(rampIterations),
	}, nil
}
