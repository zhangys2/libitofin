//! Vanna/Volga single-barrier FX engine.
//!
//! Port of `ql/experimental/barrieroption/vannavolgabarrierengine.{hpp,cpp}`
//! with [`AnalyticBarrierEngine`] as the inner Black–Scholes pricer.

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::handle::Handle;
use crate::instrument::{Instrument, InstrumentResults};
use crate::instruments::{
    BarrierArguments, BarrierOption, BarrierType, PlainVanillaPayoff, StrikedTypePayoff,
    TypePayoff, set_analytic_barrier_engine,
};
use crate::interestrate::Compounding;
use crate::math::array::Array;
use crate::math::distributions::normal::{CumulativeNormalDistribution, NormalDistribution};
use crate::math::matrix::{Matrix, inverse_3x3};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::black_formula;
use crate::pricingengines::blackdeltacalculator::BlackDeltaCalculator;
use crate::processes::BlackScholesMertonProcess;
use crate::quotes::{DeltaVolQuote, Quote, SimpleQuote};
use crate::require;
use crate::settings::Settings;
use crate::shared::{Shared, SharedMut, shared};
use crate::termstructures::volatility::BlackConstantVol;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::time::daycounters::actual365fixed::Actual365Fixed;
use crate::time::frequency::Frequency;
use crate::types::{Real, Time};

use super::vannavolgainterpolation::VannaVolgaInterpolation;

type EngineBase = GenericEngine<BarrierArguments, InstrumentResults>;

/// Vanna/Volga adjustment on top of a flat-vol single-barrier pricer.
pub struct VannaVolgaBarrierEngine {
    base: EngineBase,
    atm_vol: Shared<DeltaVolQuote>,
    vol25_put: Shared<DeltaVolQuote>,
    vol25_call: Shared<DeltaVolQuote>,
    t: Time,
    spot_fx: Handle<dyn Quote>,
    domestic_ts: Handle<dyn YieldTermStructure>,
    foreign_ts: Handle<dyn YieldTermStructure>,
    adapt_van_delta: bool,
    bs_price_with_smile: Real,
    settings: Shared<Settings<Date>>,
    normal: NormalDistribution,
    cnd: CumulativeNormalDistribution,
}

impl VannaVolgaBarrierEngine {
    #[allow(clippy::too_many_arguments)]
    #[rustfmt::skip]
    pub fn with_options(
        atm_vol: Shared<DeltaVolQuote>, vol25_put: Shared<DeltaVolQuote>,
        vol25_call: Shared<DeltaVolQuote>, spot_fx: Handle<dyn Quote>,
        domestic_ts: Handle<dyn YieldTermStructure>, foreign_ts: Handle<dyn YieldTermStructure>,
        adapt_van_delta: bool, bs_price_with_smile: Real, settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        require!(vol25_put.delta() == Some(-0.25), "25 delta put is required by vanna volga method");
        require!(vol25_call.delta() == Some(0.25), "25 delta call is required by vanna volga method");
        let t = atm_vol.maturity();
        require!(vol25_put.maturity() == vol25_call.maturity() && vol25_put.maturity() == t, "Maturity of 3 vols are not the same");
        require!(!domestic_ts.is_empty(), "domestic yield curve is not defined");
        require!(!foreign_ts.is_empty(), "foreign yield curve is not defined");
        let base = EngineBase::new(BarrierArguments::default(), InstrumentResults::default());
        base.register_with(atm_vol.observable());
        base.register_with(vol25_put.observable());
        base.register_with(vol25_call.observable());
        let observer = base.observer();
        spot_fx.register_observer(&observer);
        domestic_ts.register_observer(&observer);
        foreign_ts.register_observer(&observer);
        Ok(Self { base, atm_vol, vol25_put, vol25_call, t, spot_fx, domestic_ts, foreign_ts, adapt_van_delta, bs_price_with_smile, settings, normal: NormalDistribution::standard(), cnd: CumulativeNormalDistribution::standard() })
    }
}

impl AsObservable for VannaVolgaBarrierEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for VannaVolgaBarrierEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }
    fn results(&self) -> &dyn Results {
        self.base.results()
    }
    fn reset(&mut self) {
        self.base.reset();
    }

    #[allow(clippy::too_many_lines, clippy::similar_names)]
    #[rustfmt::skip]
    fn calculate(&mut self) -> QlResult<()> {
        const SHIFT: Real = 0.0001;
        let args = self.base.arguments();
        let barrier_type = args.barrier_type.expect("validated");
        let barrier = args.barrier.expect("validated");
        let rebate = args.rebate.expect("validated");
        let payoff = args.payoff.expect("validated");
        let exercise = Shared::clone(args.exercise.as_ref().expect("validated"));
        require!(matches!(barrier_type, BarrierType::UpIn | BarrierType::UpOut | BarrierType::DownIn | BarrierType::DownOut), "Invalid barrier type");
        let spot0 = self.spot_fx.current_link()?.value()?;
        let atm_vol0 = self.atm_vol.value()?;
        let spot_q = shared(SimpleQuote::new(spot0));
        let vol_q = shared(SimpleQuote::new(atm_vol0));
        let spot_shift = SHIFT * spot0;
        let sqrt_t = self.t.sqrt();
        let domestic = self.domestic_ts.current_link()?;
        let foreign = self.foreign_ts.current_link()?;
        let d_disc = domestic.discount(self.t, false)?;
        let f_disc = foreign.discount(self.t, false)?;
        let forward = spot0 * f_disc / d_disc;
        let eval = self.settings.evaluation_date().expect("evaluation date");
        let vol_ts = shared(BlackConstantVol::with_quote(eval, None, Handle::new(Shared::clone(&vol_q) as Shared<dyn Quote>), Actual365Fixed::new()));
        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(Shared::clone(&spot_q) as Shared<dyn Quote>),
            Handle::new(Shared::clone(&foreign) as Shared<dyn YieldTermStructure>),
            Handle::new(Shared::clone(&domestic) as Shared<dyn YieldTermStructure>),
            Handle::new(vol_ts as Shared<dyn crate::termstructures::volatility::BlackVolTermStructure>),
        ));
        let atm_strike = BlackDeltaCalculator::new(OptionType::Call, self.atm_vol.delta_type(), spot0, d_disc, f_disc, atm_vol0 * sqrt_t)?.atm_strike(self.atm_vol.atm_type())?;
        let call25_vol = self.vol25_call.value()?;
        let put25_vol = self.vol25_put.value()?;
        let put25_strike = BlackDeltaCalculator::new(OptionType::Put, self.vol25_put.delta_type(), spot0, d_disc, f_disc, put25_vol * sqrt_t)?.strike_from_delta(-0.25)?;
        let call25_strike = BlackDeltaCalculator::new(OptionType::Call, self.vol25_call.delta_type(), spot0, d_disc, f_disc, call25_vol * sqrt_t)?.strike_from_delta(0.25)?;
        let strike_vol = VannaVolgaInterpolation::new([put25_strike, atm_strike, call25_strike], [put25_vol, atm_vol0, call25_vol], spot0, d_disc, f_disc, self.t)?.value(payoff.strike())?;
        let vanilla = black_formula(payoff.option_type(), payoff.strike(), forward, strike_vol * sqrt_t, d_disc, 0.0)?;
        let van = if self.adapt_van_delta { self.bs_price_with_smile } else { vanilla };
        let up = matches!(barrier_type, BarrierType::UpIn | BarrierType::UpOut);
        let knock_out = matches!(barrier_type, BarrierType::UpOut | BarrierType::DownOut);
        if if up { spot0 >= barrier } else { spot0 <= barrier } {
            self.base.results_mut().value = Some(if knock_out { 0.0 } else { van });
            return Ok(());
        }
        let mut bump = bump_out(process, if up { BarrierType::UpOut } else { BarrierType::DownOut }, barrier, rebate, payoff, exercise, Shared::clone(&self.settings))?;
        let price_bs = bump.npv()?;
        let std_atm = atm_vol0 * sqrt_t;
        let p25c_bs = black_formula(OptionType::Call, call25_strike, forward, std_atm, d_disc, 0.0)?;
        let p25p_bs = black_formula(OptionType::Put, put25_strike, forward, std_atm, d_disc, 0.0)?;
        let p25c_mkt = black_formula(OptionType::Call, call25_strike, forward, call25_vol * sqrt_t, d_disc, 0.0)?;
        let p25p_mkt = black_formula(OptionType::Put, put25_strike, forward, put25_vol * sqrt_t, d_disc, 0.0)?;
        let g = |k| greeks(forward, k, spot0, f_disc, atm_vol0, self.t, &self.normal);
        let (vega_atm, vanna_atm, volga_atm) = g(atm_strike)?;
        let (vega25c, vanna25c, volga25c) = g(call25_strike)?;
        let (vega25p, vanna25p, volga25p) = g(put25_strike)?;
        vol_q.set_value(atm_vol0 + SHIFT);
        let vega_bar = (bump.npv()? - price_bs) / SHIFT;
        let price_bs2 = bump.npv()?;
        vol_q.set_value(atm_vol0 + 2.0 * SHIFT);
        let volga_bar = ((bump.npv()? - price_bs2) / SHIFT - vega_bar) / SHIFT;
        vol_q.set_value(atm_vol0);
        spot_q.set_value(spot0 + spot_shift);
        let d1 = bump.npv()?;
        spot_q.set_value(spot0 - spot_shift);
        let d2 = bump.npv()?;
        spot_q.set_value(spot0);
        let delta1 = (d1 - d2) / (2.0 * spot_shift);
        vol_q.set_value(atm_vol0 + SHIFT);
        spot_q.set_value(spot0 + spot_shift);
        let d1 = bump.npv()?;
        spot_q.set_value(spot0 - spot_shift);
        let d2 = bump.npv()?;
        spot_q.set_value(spot0);
        vol_q.set_value(atm_vol0);
        let vanna_bar = ((d1 - d2) / (2.0 * spot_shift) - delta1) / SHIFT;
        let mut a = Matrix::with_size(3, 3);
        a[(0, 0)] = vega_atm; a[(0, 1)] = vega25c; a[(0, 2)] = vega25p;
        a[(1, 0)] = vanna_atm; a[(1, 1)] = vanna25c; a[(1, 2)] = vanna25p;
        a[(2, 0)] = volga_atm; a[(2, 1)] = volga25c; a[(2, 2)] = volga25p;
        let q = &inverse_3x3(&a) * &Array::from([vega_bar, vanna_bar, volga_bar]);
        let zr = |ts: &dyn YieldTermStructure| ts.zero_rate(self.t, Compounding::Continuous, Frequency::NoFrequency, false).map(|z| z.rate());
        let mu = zr(domestic.as_ref())? - zr(foreign.as_ref())? - atm_vol0 * atm_vol0 / 2.0;
        let h2 = ((barrier / spot0).ln() + mu * self.t) / (atm_vol0 * sqrt_t);
        let h2p = ((spot0 / barrier).ln() + mu * self.t) / (atm_vol0 * sqrt_t);
        let tilt = (barrier / spot0).powf(2.0 * mu / (atm_vol0 * atm_vol0));
        let lambda = 1.0 - if up { self.cnd.value(h2p) + tilt * self.cnd.value(-h2) } else { self.cnd.value(-h2p) + tilt * self.cnd.value(h2) };
        let mut out_price = price_bs + lambda * (q[1] * (p25c_mkt - p25c_bs) + q[2] * (p25p_mkt - p25p_bs));
        if self.adapt_van_delta {
            out_price = (out_price + lambda * (self.bs_price_with_smile - vanilla)).max(0.0).min(self.bs_price_with_smile);
        } else {
            out_price = out_price.max(0.0).min(vanilla);
        }
        self.base.results_mut().value = Some(if knock_out { out_price } else { van - out_price });
        Ok(())
    }
}

#[rustfmt::skip]
fn greeks(forward: Real, strike: Real, spot: Real, f_disc: Real, atm_vol: Real, t: Time, normal: &NormalDistribution) -> QlResult<(Real, Real, Real)> {
    let sqrt_t = t.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * atm_vol * atm_vol * t) / (atm_vol * sqrt_t);
    let vega = spot * normal.value(d1) * sqrt_t * f_disc;
    Ok((vega, vega / spot * (1.0 - d1 / (atm_vol * sqrt_t)), vega * d1 * (d1 - atm_vol * sqrt_t) / atm_vol))
}

#[rustfmt::skip]
fn bump_out(
    process: Shared<BlackScholesMertonProcess>, ty: BarrierType, barrier: Real, rebate: Real,
    payoff: PlainVanillaPayoff, exercise: Shared<dyn Exercise>, settings: Shared<Settings<Date>>,
) -> QlResult<BarrierOption> {
    let mut option = BarrierOption::with_rebate(ty, barrier, rebate, payoff, exercise, settings)?;
    set_analytic_barrier_engine(&mut option, process);
    Ok(option)
}

#[rustfmt::skip]
pub fn set_vanna_volga_barrier_engine(option: &mut BarrierOption, engine: SharedMut<VannaVolgaBarrierEngine>) {
    option.base_mut().set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::quotes::{AtmType, DeltaType};
    use crate::shared::shared_mut;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::Month;
    use crate::types::Volatility;

    fn today() -> Date {
        Date::new(5, Month::March, 2013)
    }
    fn qh(q: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn Quote>)
    }

    /// `barrieroption.cpp` `testVannaVolgaSimpleBarrierValues` subset @ 1e-4.
    #[test]
    #[rustfmt::skip]
    fn vanna_volga_simple_barrier_subset() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let spot = shared(SimpleQuote::new(1.30265));
        let q_rate = shared(SimpleQuote::new(0.0003541));
        let r_rate = shared(SimpleQuote::new(0.0033871));
        let v25p = shared(SimpleQuote::new(0.10087));
        let vatm = shared(SimpleQuote::new(0.08925));
        let v25c = shared(SimpleQuote::new(0.08463));
        let flat = |q: &Shared<SimpleQuote>| Handle::new(shared(FlatForward::new(today(), qh(q), Actual365Fixed::new(), Compounding::Continuous, Frequency::Annual)) as Shared<dyn YieldTermStructure>);
        let r_ts = flat(&r_rate);
        let q_ts = flat(&q_rate);
        let npv = |bty, h, oty, k, t: Real, vp: Volatility, va, vc, v, s: Shared<SimpleQuote>, qr: Shared<SimpleQuote>, rr: Shared<SimpleQuote>| {
            s.set_value(1.30265); qr.set_value(if t > 1.5 { 0.0009418 } else { 0.0003541 }); rr.set_value(if t > 1.5 { 0.0039788 } else { 0.0033871 });
            v25p.set_value(vp); vatm.set_value(va); v25c.set_value(vc);
            let mut option = BarrierOption::with_rebate(bty, h, 0.0, PlainVanillaPayoff::new(oty, k), shared(EuropeanExercise::new(today() + (t * 365.0).round() as i32)), Shared::clone(&settings)).unwrap();
            let rd = r_ts.current_link().unwrap().discount(t, false).unwrap();
            let bs = black_formula(oty, k, 1.30265 * q_ts.current_link().unwrap().discount(t, false).unwrap() / rd, v * t.sqrt(), rd, 0.0).unwrap();
            set_vanna_volga_barrier_engine(&mut option, shared_mut(VannaVolgaBarrierEngine::with_options(
                shared(DeltaVolQuote::new_atm(qh(&vatm), DeltaType::Fwd, t, AtmType::DeltaNeutral)),
                shared(DeltaVolQuote::new(-0.25, qh(&v25p), t, DeltaType::Fwd)),
                shared(DeltaVolQuote::new(0.25, qh(&v25c), t, DeltaType::Fwd)),
                qh(&spot), r_ts.clone(), q_ts.clone(), true, bs, Shared::clone(&settings),
            ).unwrap()));
            option.npv().unwrap()
        };
        let cases = [
            (BarrierType::UpOut, 1.5, OptionType::Call, 1.13321, 1.0, 0.10087, 0.08925, 0.08463, 0.11638, 0.148127),
            (BarrierType::UpOut, 1.5, OptionType::Put, 1.13321, 1.0, 0.10087, 0.08925, 0.08463, 0.11638, 0.00697606),
            (BarrierType::UpIn, 1.5, OptionType::Call, 1.13321, 1.0, 0.10087, 0.08925, 0.08463, 0.11638, 0.0322202),
            (BarrierType::DownOut, 1.1, OptionType::Call, 1.13321, 1.0, 0.10087, 0.08925, 0.08463, 0.11638, 0.17746),
            (BarrierType::DownIn, 1.1, OptionType::Put, 1.13321, 1.0, 0.10087, 0.08925, 0.08463, 0.11638, 0.00732),
            (BarrierType::UpOut, 1.6, OptionType::Call, 1.06145, 2.0, 0.10891, 0.09525, 0.09197, 0.12511, 0.20493),
        ];
        for (bty, h, oty, k, t, vp, va, vc, v, expected) in cases {
            let c = npv(bty, h, oty, k, t, vp, va, vc, v, Shared::clone(&spot), Shared::clone(&q_rate), Shared::clone(&r_rate));
            assert!((c - expected).abs() <= 1e-4, "{bty:?} {oty:?} H={h}: {c} vs {expected}");
        }
    }
}
