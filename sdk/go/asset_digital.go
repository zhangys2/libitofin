package itofin

/*
#include "itofin.h"
*/
import "C"

// NewAssetOrNothingOption constructs a European asset-or-nothing option.
// Strike must be finite and nonnegative. Strictly in the money the option pays
// the terminal asset price; equality pays zero. Price uses the analytic Black
// asset-or-nothing calculator; PriceFd keeps the finite-difference path.
func (s *Session) NewAssetOrNothingOption(kind OptionType, strike float64, expiry Date, settings *Settings) (*VanillaOption, error) {
	if settings == nil {
		return nil, errNilArgument("settings")
	}
	if err := sameSession(s, settings.object); err != nil {
		return nil, err
	}
	var id C.uint64_t
	err := s.invoke(func() error {
		var e C.ItofinError
		return ffiError(C.itofin_option_asset_or_nothing_new(s.ctx, C.int32_t(kind), C.double(strike), C.int32_t(expiry.Serial()), C.uint64_t(settings.id), &id, &e), &e)
	})
	if err != nil {
		return nil, err
	}
	return &VanillaOption{object{s, uint64(id)}}, nil
}
