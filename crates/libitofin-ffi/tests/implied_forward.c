#include "itofin.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>

#define N 7

static void black_quotes(double half, double *strikes, double *cb, double *ca, double *pb, double *pa) {
    const double spot = 100.0, rate = 0.01, carry = 0.02, t = 0.5, vol = 0.25;
    const double forward = spot * exp((rate - carry) * t), df = exp(-rate * t), sd = vol * sqrt(t);
    const double ks[N] = {92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5};
    for (int i = 0; i < N; ++i) {
        double k = ks[i];
        double d1 = (log(forward / k) + 0.5 * sd * sd) / sd, d2 = d1 - sd;
        double call = df * (forward * 0.5 * erfc(-d1 / sqrt(2.0)) - k * 0.5 * erfc(-d2 / sqrt(2.0)));
        double put = df * (k * 0.5 * erfc(d2 / sqrt(2.0)) - forward * 0.5 * erfc(d1 / sqrt(2.0)));
        strikes[i] = k;
        cb[i] = call - half;
        ca[i] = call + half;
        pb[i] = put - half;
        pa[i] = put + half;
    }
}

int main(void) {
    ItofinError error = {0, {0}};
    double strikes[N], cb[N], ca[N], pb[N], pa[N];
    black_quotes(0.01, strikes, cb, ca, pb, pa);
    const double spot = 100.0, df = exp(-0.01 * 0.5);

    ItofinImpliedForwardOptions options;
    assert(itofin_implied_forward_options_default(&options, &error) == 0);
    assert(options.level == 0 && options.dividend_count == 0 && options.has_atm_vol == 0);
    assert(options.max_accurate_prices == 64);
    assert(fabs(options.carry_flag_high - 0.15) < 1e-15);
    assert(itofin_implied_forward_options_default(NULL, &error) == ITOFIN_INVALID_ARGUMENT);

    ItofinImpliedForwardResultEx out;
    /* Model path, fast level. */
    assert(itofin_implied_forward_calculate_with_options(strikes, cb, ca, pb, pa, N, 2, &spot, &df, 0.5,
                                                         &options, &out, &error) == 0);
    assert(out.tier_code == 1 && out.flags == 0 && out.has_model == 1);
    assert(isfinite(out.solved_carry) && fabs(out.solved_carry - 0.02) < 0.02);
    assert(fabs(out.call_put_iv_gap) < 1e-6 && out.iterations >= 1 && out.accurate_prices_used == 0);
    assert(out.total_pairs_received == N && out.min_spread > 0.0 && out.core.forward > 0.0);

    /* Accurate level spends two accurate prices per strike. */
    options.level = 2;
    assert(itofin_implied_forward_calculate_with_options(strikes, cb, ca, pb, pa, N, 2, &spot, &df, 0.5,
                                                         &options, &out, &error) == 0);
    assert(out.tier_code == 2 && out.accurate_prices_used == 2 * N);

    /* A schedule at the fast level is ignored and flagged. */
    const double times[1] = {0.2}, amounts[1] = {0.5};
    options.level = 1;
    options.dividend_times = times;
    options.dividend_amounts = amounts;
    options.dividend_count = 1;
    assert(itofin_implied_forward_calculate_with_options(strikes, cb, ca, pb, pa, N, 2, &spot, &df, 0.5,
                                                         &options, &out, &error) == 0);
    assert(out.tier_code == 1 && out.flags == 1);

    /* No discount factor: the corridor tier, no model outputs. */
    assert(itofin_implied_forward_options_default(&options, &error) == 0);
    assert(itofin_implied_forward_calculate_with_options(strikes, cb, ca, pb, pa, N, 2, &spot, NULL, 0.5,
                                                         &options, &out, &error) == 0);
    assert(out.tier_code == 0 && out.has_model == 0 && isnan(out.solved_carry));

    /* Rejections leave the output untouched. */
    out.tier_code = -7;
    options.level = 9;
    assert(itofin_implied_forward_calculate_with_options(strikes, cb, ca, pb, pa, N, 2, &spot, &df, 0.5,
                                                         &options, &out, &error) == ITOFIN_INVALID_ARGUMENT);
    assert(out.tier_code == -7);
    assert(itofin_implied_forward_calculate_with_options(strikes, cb, ca, pb, pa, N, 2, &spot, &df, 0.5,
                                                         NULL, &out, &error) == ITOFIN_INVALID_ARGUMENT);

    /* The original entry point is unchanged. */
    ItofinImpliedForwardResult plain;
    assert(itofin_implied_forward_calculate(strikes, cb, ca, pb, pa, N, 2, &spot, &df, 0.5, &plain, &error) == 0);
    assert(plain.forward > 0.0);
    return 0;
}
