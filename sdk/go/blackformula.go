package itofin

/*
#include "itofin.h"
*/
import "C"

// BlackFormulaImpliedStdDev returns the total Black standard deviation.
// optionType is 1 for a call and -1 for a put. A nil guess uses the core approximation.
func BlackFormulaImpliedStdDev(optionType int32, strike, forward, blackPrice, discount, displacement float64, guess *float64, accuracy float64, maxIterations int64) (float64, error) {
	var out C.double
	var e C.ItofinError
	var guessPtr *C.double
	var guessValue C.double
	if guess != nil {
		guessValue = C.double(*guess)
		guessPtr = &guessValue
	}
	err := ffiError(C.itofin_black_formula_implied_std_dev(
		C.int32_t(optionType),
		C.double(strike),
		C.double(forward),
		C.double(blackPrice),
		C.double(discount),
		C.double(displacement),
		guessPtr,
		C.double(accuracy),
		C.int64_t(maxIterations),
		&out,
		&e,
	), &e)
	return float64(out), err
}

// BlackFormulaImpliedVolatility returns annualized Black volatility as a decimal.
// optionType is 1 for a call and -1 for a put. expiry is a positive year fraction.
func BlackFormulaImpliedVolatility(optionType int32, strike, forward, expiry, blackPrice, discount, displacement, accuracy float64, maxIterations int64) (float64, error) {
	var out C.double
	var e C.ItofinError
	err := ffiError(C.itofin_black_formula_implied_volatility(
		C.int32_t(optionType),
		C.double(strike),
		C.double(forward),
		C.double(expiry),
		C.double(blackPrice),
		C.double(discount),
		C.double(displacement),
		C.double(accuracy),
		C.int64_t(maxIterations),
		&out,
		&e,
	), &e)
	return float64(out), err
}
