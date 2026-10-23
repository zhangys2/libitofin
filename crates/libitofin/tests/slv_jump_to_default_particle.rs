//! End-to-end checks for stochastic local volatility with jump to default.
//!
//! Algebraic identities (survival, martingale, parity, greeks signs, the EDS
//! limit) live next to the modules. This file checks the statistical ones:
//! repricing a constant-hazard flat-vol market, a skewed surface, and the
//! zero-mixing finite-difference engine.

use libitofin::errors::QlResult;
use libitofin::exercise::{EuropeanExercise, Exercise};
use libitofin::handle::Handle;
use libitofin::instruments::{PlainVanillaPayoff, StrikedTypePayoff};
use libitofin::interestrate::Compounding;
use libitofin::methods::finitedifferences::solvers::FdmSchemeDesc;
use libitofin::models::equity::jumptodefault::{
    calibrate_slv_jump_to_default, MomentGrid, SlvJtdParticleConfig,
};
use libitofin::option::OptionType;
use libitofin::patterns::observable::{AsObservable, Observable};
use libitofin::pricingengines::blackformula::{black_formula, black_formula_implied_std_dev};
use libitofin::pricingengines::vanilla::{
    FdJumpToDefaultVanillaEngine, McSlvJumpToDefaultVanillaEngine,
};
use libitofin::processes::{GeneralizedBlackScholesProcess, HestonProcess};
use libitofin::quotes::{Quote, SimpleQuote};
use libitofin::shared::{shared, Shared};
use libitofin::termstructures::credit::defaulttermstructure::DefaultProbabilityTermStructure;
use libitofin::termstructures::credit::flathazardrate::FlatHazardRate;
use libitofin::termstructures::credit::interpolatedhazardratecurve::InterpolatedHazardRateCurve;
use libitofin::termstructures::volatility::{
    BlackConstantVol, BlackVolTermStructure, LocalVolTermStructure, VolatilityTermStructure,
};
use libitofin::termstructures::yields::FlatForward;
use libitofin::termstructures::yieldtermstructure::YieldTermStructure;
use libitofin::termstructures::{TermStructure, TermStructureBase};
use libitofin::time::businessdayconvention::BusinessDayConvention;
use libitofin::time::date::{Date, Month};
use libitofin::time::daycounters::actual365fixed::Actual365Fixed;
use libitofin::time::frequency::Frequency;
use libitofin::types::{Natural, Rate, Real, Time, Volatility};

fn today() -> Date {
    Date::new(15, Month::June, 2026)
}

fn yield_curve(rate: Real) -> Handle<dyn YieldTermStructure> {
    Handle::new(shared(FlatForward::with_rate(
        today(),
        rate,
        Actual365Fixed::new(),
        Compounding::Continuous,
        Frequency::Annual,
    )) as Shared<dyn YieldTermStructure>)
}

fn flat_black(vol: Volatility) -> Handle<dyn BlackVolTermStructure> {
    Handle::new(shared(BlackConstantVol::new(
        today(),
        None,
        vol,
        Actual365Fixed::new(),
    )) as Shared<dyn BlackVolTermStructure>)
}

fn flat_hazard(hazard: Real) -> Handle<dyn DefaultProbabilityTermStructure> {
    Handle::new(shared(FlatHazardRate::with_rate(
        today(),
        hazard,
        Actual365Fixed::new(),
    )) as Shared<dyn DefaultProbabilityTermStructure>)
}

fn heston(
    rate: Real,
    dividend: Real,
    spot: Real,
    kappa: Real,
    sigma: Real,
    rho: Real,
) -> Shared<HestonProcess> {
    shared(HestonProcess::new(
        yield_curve(rate),
        yield_curve(dividend),
        Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
        1.0,
        kappa,
        1.0,
        sigma,
        rho,
    ))
}

fn accurate_config(particles: usize, steps: usize) -> SlvJtdParticleConfig {
    SlvJtdParticleConfig {
        n_particles: particles,
        time_steps_per_year: steps,
        n_strike_knots: 31,
        moment_grid: MomentGrid::new(8.0, 32).unwrap(),
        ..SlvJtdParticleConfig::default()
    }
}

fn implied_vol(
    option_type: OptionType,
    price: Real,
    spot: Real,
    strike: Real,
    expiry: Time,
    rate: Real,
    dividend: Real,
) -> QlResult<Volatility> {
    let forward = spot * ((rate - dividend) * expiry).exp();
    let discount = (-rate * expiry).exp();
    let std = black_formula_implied_std_dev(
        option_type,
        strike,
        forward,
        price,
        discount,
        0.0,
        Some(0.25 * expiry.sqrt()),
        1.0e-10,
        80 as Natural,
    )?;
    Ok(std / expiry.sqrt())
}

fn rmse(errors: &[Real]) -> Real {
    (errors.iter().map(|error| error * error).sum::<Real>() / errors.len() as Real).sqrt()
}

/// Black vols of the constant-hazard, flat-diffusion call prices.
struct ConstantHazardBlackVol {
    base: TermStructureBase,
    spot: Real,
    rate: Real,
    dividend: Real,
    volatility: Real,
    hazard: Real,
}

impl ConstantHazardBlackVol {
    fn price(&self, strike: Real, expiry: Time) -> QlResult<Real> {
        let time = expiry.max(1.0e-6);
        let discount = (-self.rate * time).exp() * (-self.hazard * time).exp();
        let forward = self.spot * ((self.rate - self.dividend + self.hazard) * time).exp();
        black_formula(
            OptionType::Call,
            strike,
            forward,
            self.volatility * time.sqrt(),
            discount,
            0.0,
        )
    }
}

impl AsObservable for ConstantHazardBlackVol {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl TermStructure for ConstantHazardBlackVol {
    fn base(&self) -> &TermStructureBase {
        &self.base
    }

    fn max_date(&self) -> Date {
        Date::max_date()
    }
}

impl VolatilityTermStructure for ConstantHazardBlackVol {
    fn business_day_convention(&self) -> BusinessDayConvention {
        BusinessDayConvention::Following
    }

    fn min_strike(&self) -> Rate {
        Rate::MIN
    }

    fn max_strike(&self) -> Rate {
        Rate::MAX
    }
}

impl BlackVolTermStructure for ConstantHazardBlackVol {
    fn black_vol_impl(&self, t: Time, strike: Real) -> QlResult<Volatility> {
        let variance = self.black_variance_impl(t, strike)?;
        let time = if t == 0.0 { 1.0e-8 } else { t };
        Ok((variance / time).sqrt())
    }

    fn black_variance_impl(&self, t: Time, strike: Real) -> QlResult<Real> {
        let time = if t == 0.0 { 1.0e-6 } else { t };
        let forward = self.spot * ((self.rate - self.dividend) * time).exp();
        let price = self.price(strike, time)?;
        let discount = (-self.rate * time).exp();
        let std = black_formula_implied_std_dev(
            OptionType::Call,
            strike,
            forward,
            price,
            discount,
            0.0,
            Some(self.volatility * time.sqrt()),
            1.0e-12,
            80 as Natural,
        )?;
        Ok(std * std * t / time)
    }
}

/// Raw SVI total variance, the skewed market of the joint-repricing test.
struct SviBlackVol {
    base: TermStructureBase,
    spot: Real,
    rate: Real,
    dividend: Real,
    base_a: Real,
    base_b: Real,
    rho: Real,
    shift: Real,
    sigma: Real,
    term_slope: Real,
}

impl SviBlackVol {
    fn total_variance(&self, strike: Real, expiry: Time) -> Real {
        let time = expiry.max(1.0e-8);
        let forward = self.spot * ((self.rate - self.dividend) * time).exp();
        let log_moneyness = (strike / forward).ln();
        let centered = log_moneyness - self.shift;
        let radius = (centered * centered + self.sigma * self.sigma).sqrt();
        let core = self.rho * centered + radius;
        time * (self.base_a + self.term_slope * time) + time.powf(0.85) * self.base_b * core
    }
}

impl AsObservable for SviBlackVol {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl TermStructure for SviBlackVol {
    fn base(&self) -> &TermStructureBase {
        &self.base
    }

    fn max_date(&self) -> Date {
        Date::max_date()
    }
}

impl VolatilityTermStructure for SviBlackVol {
    fn business_day_convention(&self) -> BusinessDayConvention {
        BusinessDayConvention::Following
    }

    fn min_strike(&self) -> Rate {
        Rate::MIN
    }

    fn max_strike(&self) -> Rate {
        Rate::MAX
    }
}

impl BlackVolTermStructure for SviBlackVol {
    fn black_vol_impl(&self, t: Time, strike: Real) -> QlResult<Volatility> {
        let time = if t == 0.0 { 1.0e-8 } else { t };
        Ok((self.total_variance(strike, time) / time).sqrt())
    }

    fn black_variance_impl(&self, t: Time, strike: Real) -> QlResult<Real> {
        let time = if t == 0.0 { 1.0e-8 } else { t };
        Ok(self.total_variance(strike, time) * t / time)
    }
}

fn market_iv(surface: &dyn BlackVolTermStructure, strike: Real, expiry: Time) -> Volatility {
    surface.black_vol(expiry, strike, true).unwrap()
}

#[test]
fn constant_hazard_flat_vol_is_repriced_within_twenty_bps() {
    let spot = 100.0;
    let rate = 0.02;
    let dividend = 0.0;
    let vol = 0.25;
    let hazard = 0.03;
    let black = Handle::new(shared(ConstantHazardBlackVol {
        base: TermStructureBase::with_reference_date(today(), None, Some(Actual365Fixed::new())),
        spot,
        rate,
        dividend,
        volatility: vol,
        hazard,
    }) as Shared<dyn BlackVolTermStructure>);
    let calibrated = calibrate_slv_jump_to_default(
        black,
        heston(rate, dividend, spot, 1.8, 0.35, -0.4),
        flat_hazard(hazard),
        0.0,
        1.0,
        1.0,
        &accurate_config(20_000, 150),
    )
    .unwrap();
    let engine = McSlvJumpToDefaultVanillaEngine::new(shared(calibrated));
    let mut errors = Vec::new();
    for expiry in [0.5, 1.0] {
        let forward = spot * ((rate - dividend) * expiry).exp();
        for moneyness in [0.80, 0.90, 1.00, 1.10, 1.20] {
            let strike = moneyness * forward;
            let (price, _, _) = engine
                .price_components(OptionType::Call, strike, expiry)
                .unwrap();
            let oracle = {
                let discount = (-rate * expiry).exp() * (-hazard * expiry).exp();
                let jumped_forward = spot * ((rate - dividend + hazard) * expiry).exp();
                black_formula(
                    OptionType::Call,
                    strike,
                    jumped_forward,
                    vol * expiry.sqrt(),
                    discount,
                    0.0,
                )
                .unwrap()
            };
            let iv_mc = implied_vol(
                OptionType::Call,
                price,
                spot,
                strike,
                expiry,
                rate,
                dividend,
            )
            .unwrap();
            let iv_oracle = implied_vol(
                OptionType::Call,
                oracle,
                spot,
                strike,
                expiry,
                rate,
                dividend,
            )
            .unwrap();
            errors.push(iv_mc - iv_oracle);
        }
    }
    let error = rmse(&errors);
    assert!(
        error <= 0.0020,
        "constant-hazard IV RMSE {error} exceeds 20 bps; errors {errors:?}"
    );
}

#[test]
fn skewed_surface_and_upward_cds_are_jointly_repriced() {
    let spot = 100.0;
    let rate = 0.04;
    let dividend = 0.015;
    let svi = shared(SviBlackVol {
        base: TermStructureBase::with_reference_date(today(), None, Some(Actual365Fixed::new())),
        spot,
        rate,
        dividend,
        base_a: 0.034,
        base_b: 0.110,
        rho: -0.70,
        shift: 0.02,
        sigma: 0.20,
        term_slope: 0.004,
    });
    let black = Handle::new(Shared::clone(&svi) as Shared<dyn BlackVolTermStructure>);
    // Piecewise hazards near par spreads 55, 75, 92 and 110 bps at recovery 40%.
    let credit = InterpolatedHazardRateCurve::new(
        vec![
            today(),
            today() + 183,
            today() + 365,
            today() + 548,
            today() + 730,
        ],
        vec![0.0092, 0.0125, 0.0153, 0.0183, 0.0183],
        Actual365Fixed::new(),
        libitofin::math::interpolations::flat::BackwardFlat,
    )
    .unwrap();
    credit.enable_extrapolation();
    let survival = Handle::new(shared(credit) as Shared<dyn DefaultProbabilityTermStructure>);
    let calibrated = calibrate_slv_jump_to_default(
        black,
        heston(rate, dividend, spot, 2.2, 0.45, -0.55),
        survival,
        -1.25,
        1.0,
        1.5,
        &{
            let mut config = accurate_config(24_000, 280);
            config.n_strike_knots = 61;
            config.kde_bandwidth_scale = 0.50;
            config
        },
    )
    .unwrap();
    let engine = McSlvJumpToDefaultVanillaEngine::new(shared(calibrated));
    let mut errors = Vec::new();
    for expiry in [0.5, 1.0, 1.5] {
        for moneyness in [0.80, 0.90, 1.00, 1.10, 1.20] {
            let strike = moneyness * spot;
            let (price, _, _) = engine
                .price_components(OptionType::Call, strike, expiry)
                .unwrap();
            let iv_mc = implied_vol(
                OptionType::Call,
                price,
                spot,
                strike,
                expiry,
                rate,
                dividend,
            )
            .unwrap();
            errors.push(iv_mc - market_iv(&*svi, strike, expiry));
        }
    }
    let mut worst_atm: Real = 0.0;
    let mut atm_gaps = Vec::new();
    for expiry in [0.5, 1.0, 1.5] {
        let (price, _, _) = engine
            .price_components(OptionType::Call, spot, expiry)
            .unwrap();
        let iv_mc =
            implied_vol(OptionType::Call, price, spot, spot, expiry, rate, dividend).unwrap();
        let gap = (iv_mc - market_iv(&*svi, spot, expiry)).abs();
        atm_gaps.push((expiry, gap));
        worst_atm = worst_atm.max(gap);
    }
    let error = rmse(&errors);
    assert!(
        error <= 0.0025,
        "skewed-surface IV RMSE {error} exceeds 25 bps; errors {errors:?}"
    );
    assert!(
        worst_atm <= 0.0010,
        "ATM IV error {worst_atm} exceeds 10 bps; gaps {atm_gaps:?}"
    );
}

#[test]
fn zero_mixing_matches_the_finite_difference_engine() {
    let spot = 100.0;
    let rate = 0.03;
    let dividend = 0.01;
    let hazard = 0.02;
    let calibrated = calibrate_slv_jump_to_default(
        flat_black(0.25),
        heston(rate, dividend, spot, 1.5, 0.30, -0.3),
        flat_hazard(hazard),
        0.0,
        0.0,
        1.0,
        &accurate_config(6_000, 80),
    )
    .unwrap();
    let mc = McSlvJumpToDefaultVanillaEngine::new(shared(calibrated));
    let local_vol = Handle::new(
        Shared::clone(&mc.calibration().leverage_surface) as Shared<dyn LocalVolTermStructure>
    );
    let process = shared(GeneralizedBlackScholesProcess::new(
        Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
        yield_curve(dividend),
        yield_curve(rate),
        flat_black(0.25),
    ));
    let mut fd = FdJumpToDefaultVanillaEngine::new(
        process,
        Some(local_vol),
        Shared::clone(&mc.calibration().calibrated_hazard),
        80,
        120,
        FdmSchemeDesc::douglas(),
    );
    let expiry = today() + 365;
    for (option_type, strike) in [
        (OptionType::Call, 80.0),
        (OptionType::Call, 100.0),
        (OptionType::Call, 120.0),
        (OptionType::Put, 80.0),
        (OptionType::Put, 100.0),
        (OptionType::Put, 120.0),
    ] {
        let (mc_price, _, _) = mc.price_components(option_type, strike, 1.0).unwrap();
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(option_type, strike));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));
        let fd_price = fd.price(Shared::clone(&payoff), exercise).unwrap();
        let iv_mc = implied_vol(option_type, mc_price, spot, strike, 1.0, rate, dividend).unwrap();
        let iv_fd = implied_vol(option_type, fd_price, spot, strike, 1.0, rate, dividend).unwrap();
        assert!(
            (iv_mc - iv_fd).abs() <= 0.0025,
            "{option_type} K={strike} MC IV {iv_mc} FD IV {iv_fd} prices {mc_price} vs {fd_price}"
        );
    }
}
