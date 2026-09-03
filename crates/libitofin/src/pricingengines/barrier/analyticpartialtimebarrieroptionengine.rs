//! Analytic partial-time barrier option engine.
//!
//! Port of `ql/pricingengines/barrier/analyticpartialtimebarrieroptionengine.{hpp,cpp}`.

use crate::errors::QlResult;
use crate::fail;
use crate::instrument::Instrument;
use crate::instruments::{
    BarrierType, PartialBarrierRange, PartialTimeBarrierArguments, PartialTimeBarrierResults,
    PlainVanillaPayoff, StrikedTypePayoff, TypePayoff,
};
use crate::interestrate::Compounding;
use crate::math::distributions::bivariatenormal::BivariateCumulativeNormalDistributionDr78;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::BlackCalculator;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::frequency::Frequency;
use crate::types::{Rate, Real, Time, Volatility};

type EngineBase = GenericEngine<PartialTimeBarrierArguments, PartialTimeBarrierResults>;

/// Pricing engine for European partial-time barrier options.
pub struct AnalyticPartialTimeBarrierOptionEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl AnalyticPartialTimeBarrierOptionEngine {
    /// `AnalyticPartialTimeBarrierOptionEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(
            PartialTimeBarrierArguments::default(),
            PartialTimeBarrierResults::default(),
        );
        base.register_with(process.observable());
        Self { base, process }
    }

    fn underlying(&self) -> QlResult<Real> {
        self.process.x0()
    }

    fn residual_time(&self) -> QlResult<Time> {
        let exercise = self.base.arguments().exercise.as_ref().expect("validated");
        StochasticProcess1D::time(&*self.process, &exercise.last_date())
    }

    fn cover_event_time(&self) -> QlResult<Time> {
        let cover = self.base.arguments().cover_event_date.expect("validated");
        StochasticProcess1D::time(&*self.process, &cover)
    }

    fn volatility(&self, t: Time, strike: Real) -> QlResult<Volatility> {
        let vol_ts = self.process.black_volatility().current_link()?;
        vol_ts.black_vol(t, strike, true)
    }

    fn risk_free_rate(&self, process: &GeneralizedBlackScholesProcess) -> QlResult<Rate> {
        let t = self.residual_time()?;
        let r_ts = process.risk_free_rate().current_link()?;
        Ok(r_ts
            .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate())
    }

    fn dividend_yield(&self, process: &GeneralizedBlackScholesProcess) -> QlResult<Rate> {
        let t = self.residual_time()?;
        let q_ts = process.dividend_yield().current_link()?;
        Ok(q_ts
            .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate())
    }

    fn rho(&self) -> QlResult<Real> {
        Ok((self.cover_event_time()? / self.residual_time()?).sqrt())
    }

    fn mu(&self, strike: Real, b: Rate) -> QlResult<Real> {
        let vol = self.volatility(self.cover_event_time()?, strike)?;
        Ok((b - vol * vol / 2.0) / (vol * vol))
    }

    fn d1(&self, strike: Real, b: Rate) -> QlResult<Real> {
        let t2 = self.residual_time()?;
        let vol = self.volatility(t2, strike)?;
        let s = self.underlying()?;
        Ok(((s / strike).ln() + (b + vol * vol / 2.0) * t2) / (t2.sqrt() * vol))
    }

    fn d2(&self, strike: Real, b: Rate) -> QlResult<Real> {
        let t2 = self.residual_time()?;
        Ok(self.d1(strike, b)? - self.volatility(t2, strike)? * t2.sqrt())
    }

    fn e1(&self, barrier: Real, strike: Real, b: Rate) -> QlResult<Real> {
        let t1 = self.cover_event_time()?;
        let vol = self.volatility(t1, strike)?;
        let s = self.underlying()?;
        Ok(((s / barrier).ln() + (b + vol * vol / 2.0) * t1) / (t1.sqrt() * vol))
    }

    fn e2(&self, barrier: Real, strike: Real, b: Rate) -> QlResult<Real> {
        let t1 = self.cover_event_time()?;
        Ok(self.e1(barrier, strike, b)? - self.volatility(t1, strike)? * t1.sqrt())
    }

    fn e3(&self, barrier: Real, strike: Real, b: Rate) -> QlResult<Real> {
        let t1 = self.cover_event_time()?;
        let vol = self.volatility(t1, strike)?;
        let s = self.underlying()?;
        Ok(self.e1(barrier, strike, b)? + 2.0 * (barrier / s).ln() / (vol * t1.sqrt()))
    }

    fn e4(&self, barrier: Real, strike: Real, b: Rate) -> QlResult<Real> {
        let t = self.cover_event_time()?;
        Ok(self.e3(barrier, strike, b)? - self.volatility(t, strike)? * t.sqrt())
    }

    fn f1(&self, barrier: Real, strike: Real, b: Rate) -> QlResult<Real> {
        let s = self.underlying()?;
        let t = self.residual_time()?;
        let sigma = self.volatility(t, strike)?;
        let num = (s / strike).ln() + 2.0 * (barrier / s).ln() + (b + sigma * sigma / 2.0) * t;
        Ok(num / (sigma * t.sqrt()))
    }

    fn f2(&self, barrier: Real, strike: Real, b: Rate) -> QlResult<Real> {
        let t = self.residual_time()?;
        Ok(self.f1(barrier, strike, b)? - self.volatility(t, strike)? * t.sqrt())
    }

    fn g1(&self, barrier: Real, strike: Real, b: Rate) -> QlResult<Real> {
        let t2 = self.residual_time()?;
        let vol = self.volatility(t2, strike)?;
        let s = self.underlying()?;
        Ok(((s / barrier).ln() + (b + vol * vol / 2.0) * t2) / (t2.sqrt() * vol))
    }

    fn g2(&self, barrier: Real, strike: Real, b: Rate) -> QlResult<Real> {
        let t2 = self.residual_time()?;
        Ok(self.g1(barrier, strike, b)? - self.volatility(t2, strike)? * t2.sqrt())
    }

    fn g3(&self, barrier: Real, strike: Real, b: Rate) -> QlResult<Real> {
        let t2 = self.residual_time()?;
        let vol = self.volatility(t2, strike)?;
        let s = self.underlying()?;
        Ok(self.g1(barrier, strike, b)? + 2.0 * (barrier / s).ln() / (vol * t2.sqrt()))
    }

    fn g4(&self, barrier: Real, strike: Real, b: Rate) -> QlResult<Real> {
        let t2 = self.residual_time()?;
        Ok(self.g3(barrier, strike, b)? - self.volatility(t2, strike)? * t2.sqrt())
    }

    fn hs(&self, s: Real, h: Real, power: Real) -> Real {
        (h / s).powf(power)
    }

    fn m(&self, a: Real, b: Real, rho: Real) -> QlResult<Real> {
        Ok(BivariateCumulativeNormalDistributionDr78::new(rho)?.value(a, b))
    }

    fn calculate_with(
        &self,
        arguments: &PartialTimeBarrierArguments,
        payoff: &PlainVanillaPayoff,
        process: &GeneralizedBlackScholesProcess,
    ) -> QlResult<Real> {
        let barrier_type = arguments.barrier_type.expect("validated");
        let barrier_range = arguments.barrier_range.expect("validated");
        let r = self.risk_free_rate(process)?;
        let q = self.dividend_yield(process)?;
        let barrier = arguments.barrier.expect("validated");
        let strike = payoff.strike();

        match barrier_type {
            BarrierType::DownOut => match barrier_range {
                PartialBarrierRange::Start => self.ca(1, barrier, strike, r, q),
                PartialBarrierRange::EndB1 => self.co_b1(barrier, strike, r, q),
                PartialBarrierRange::EndB2 => {
                    self.co_b2(BarrierType::DownOut, barrier, strike, r, q)
                }
            },
            BarrierType::DownIn => match barrier_range {
                PartialBarrierRange::Start => self.cia(1, barrier, strike, r, q),
                PartialBarrierRange::EndB1 | PartialBarrierRange::EndB2 => {
                    fail!("Down-and-in partial-time end barrier is not implemented")
                }
            },
            BarrierType::UpOut => match barrier_range {
                PartialBarrierRange::Start => self.ca(-1, barrier, strike, r, q),
                PartialBarrierRange::EndB1 => self.co_b1(barrier, strike, r, q),
                PartialBarrierRange::EndB2 => self.co_b2(BarrierType::UpOut, barrier, strike, r, q),
            },
            BarrierType::UpIn => match barrier_range {
                PartialBarrierRange::Start => self.cia(-1, barrier, strike, r, q),
                PartialBarrierRange::EndB1 | PartialBarrierRange::EndB2 => {
                    fail!("Up-and-in partial-time end barrier is not implemented")
                }
            },
        }
    }

    fn co_b2(
        &self,
        barrier_type: BarrierType,
        barrier: Real,
        strike: Real,
        r: Rate,
        q: Rate,
    ) -> QlResult<Real> {
        require!(
            strike < barrier,
            "case of strike>barrier is not implemented for OutEnd B2 type"
        );
        let b = r - q;
        let t = self.residual_time()?;
        let s = self.underlying()?;
        let mu_ = self.mu(strike, b)?;
        let g1_ = self.g1(barrier, strike, b)?;
        let g2_ = self.g2(barrier, strike, b)?;
        let g3_ = self.g3(barrier, strike, b)?;
        let g4_ = self.g4(barrier, strike, b)?;
        let e1_ = self.e1(barrier, strike, b)?;
        let e2_ = self.e2(barrier, strike, b)?;
        let e3_ = self.e3(barrier, strike, b)?;
        let e4_ = self.e4(barrier, strike, b)?;
        let rho_ = self.rho()?;
        let hs_mu = self.hs(s, barrier, 2.0 * mu_);
        let hs_mu1 = self.hs(s, barrier, 2.0 * (mu_ + 1.0));
        let x1 = strike * (-r * t).exp();

        match barrier_type {
            BarrierType::DownOut => {
                let mut result = s * ((b - r) * t).exp();
                result *= self.m(g1_, e1_, rho_)? - hs_mu1 * self.m(g3_, -e3_, -rho_)?;
                result -= x1 * (self.m(g2_, e2_, rho_)? - hs_mu * self.m(g4_, -e4_, -rho_)?);
                Ok(result)
            }
            BarrierType::UpOut => {
                let mut result = s * ((b - r) * t).exp();
                result *= self.m(-g1_, -e1_, rho_)? - hs_mu1 * self.m(-g3_, e3_, -rho_)?;
                result -= x1 * (self.m(-g2_, -e2_, rho_)? - hs_mu * self.m(-g4_, e4_, -rho_)?);
                result -= s
                    * ((b - r) * t).exp()
                    * (self.m(-self.d1(strike, b)?, -e1_, rho_)?
                        - hs_mu1 * self.m(e3_, -self.f1(barrier, strike, b)?, -rho_)?);
                result += x1
                    * (self.m(-self.d2(strike, b)?, -e2_, rho_)?
                        - hs_mu * self.m(e4_, -self.f2(barrier, strike, b)?, -rho_)?);
                Ok(result)
            }
            _ => fail!("invalid barrier type"),
        }
    }

    fn co_b1(&self, barrier: Real, strike: Real, r: Rate, q: Rate) -> QlResult<Real> {
        let b = r - q;
        let t = self.residual_time()?;
        let s = self.underlying()?;
        let mu_ = self.mu(strike, b)?;
        let g1_ = self.g1(barrier, strike, b)?;
        let g2_ = self.g2(barrier, strike, b)?;
        let g3_ = self.g3(barrier, strike, b)?;
        let g4_ = self.g4(barrier, strike, b)?;
        let e1_ = self.e1(barrier, strike, b)?;
        let e2_ = self.e2(barrier, strike, b)?;
        let e3_ = self.e3(barrier, strike, b)?;
        let e4_ = self.e4(barrier, strike, b)?;
        let rho_ = self.rho()?;
        let hs_mu = self.hs(s, barrier, 2.0 * mu_);
        let hs_mu1 = self.hs(s, barrier, 2.0 * (mu_ + 1.0));
        let x1 = strike * (-r * t).exp();

        if strike > barrier {
            let mut result = s * ((b - r) * t).exp();
            result *= self.m(self.d1(strike, b)?, e1_, rho_)?
                - hs_mu1 * self.m(self.f1(barrier, strike, b)?, -e3_, -rho_)?;
            result -= x1
                * (self.m(self.d2(strike, b)?, e2_, rho_)?
                    - hs_mu * self.m(self.f2(barrier, strike, b)?, -e4_, -rho_)?);
            Ok(result)
        } else {
            let s1 = s * ((b - r) * t).exp();
            let mut result = s1;
            result *= self.m(-g1_, -e1_, rho_)? - hs_mu1 * self.m(-g3_, e3_, -rho_)?;
            result -= x1 * (self.m(-g2_, -e2_, rho_)? - hs_mu * self.m(-g4_, e4_, -rho_)?);
            result -= s1
                * (self.m(-self.d1(strike, b)?, -e1_, rho_)?
                    - hs_mu1 * self.m(-self.f1(barrier, strike, b)?, e3_, -rho_)?);
            result += x1
                * (self.m(-self.d2(strike, b)?, -e2_, rho_)?
                    - hs_mu * self.m(-self.f2(barrier, strike, b)?, e4_, -rho_)?);
            result += s1 * (self.m(g1_, e1_, rho_)? - hs_mu1 * self.m(g3_, -e3_, -rho_)?);
            result -= x1 * (self.m(g2_, e2_, rho_)? - hs_mu * self.m(g4_, -e4_, -rho_)?);
            Ok(result)
        }
    }

    fn cia(&self, eta: i32, barrier: Real, strike: Real, r: Rate, q: Rate) -> QlResult<Real> {
        let payoff = self.base.arguments().payoff.expect("validated");
        let t = self.residual_time()?;
        let s = self.underlying()?;
        let vol = self.volatility(t, strike)?;
        let df = (-r * t).exp();
        let forward = s * (-q * t).exp() / df;
        let black = BlackCalculator::with_striked_payoff(
            &payoff as &dyn StrikedTypePayoff,
            forward,
            vol * t.sqrt(),
            df,
        )?;
        Ok(black.value().max(0.0) - self.ca(eta, barrier, strike, r, q)?)
    }

    fn ca(&self, eta: i32, barrier: Real, strike: Real, r: Rate, q: Rate) -> QlResult<Real> {
        let b = r - q;
        let rho_ = self.rho()?;
        let t = self.residual_time()?;
        let s = self.underlying()?;
        let mu_ = self.mu(strike, b)?;
        let e1_ = self.e1(barrier, strike, b)?;
        let e2_ = self.e2(barrier, strike, b)?;
        let e3_ = self.e3(barrier, strike, b)?;
        let e4_ = self.e4(barrier, strike, b)?;
        let hs_mu = self.hs(s, barrier, 2.0 * mu_);
        let hs_mu1 = self.hs(s, barrier, 2.0 * (mu_ + 1.0));
        let eta_f = f64::from(eta);

        let mut result = s * ((b - r) * t).exp();
        result *= self.m(self.d1(strike, b)?, eta_f * e1_, eta_f * rho_)?
            - hs_mu1 * self.m(self.f1(barrier, strike, b)?, eta_f * e3_, eta_f * rho_)?;
        result -= strike
            * (-r * t).exp()
            * (self.m(self.d2(strike, b)?, eta_f * e2_, eta_f * rho_)?
                - hs_mu * self.m(self.f2(barrier, strike, b)?, eta_f * e4_, eta_f * rho_)?);
        Ok(result)
    }

    fn symmetric_barrier_type(barrier_type: BarrierType) -> BarrierType {
        match barrier_type {
            BarrierType::UpIn => BarrierType::DownIn,
            BarrierType::DownIn => BarrierType::UpIn,
            BarrierType::UpOut => BarrierType::DownOut,
            BarrierType::DownOut => BarrierType::UpOut,
        }
    }

    fn swapped_process(&self) -> GeneralizedBlackScholesProcess {
        GeneralizedBlackScholesProcess::new(
            self.process.state_variable(),
            self.process.risk_free_rate(),
            self.process.dividend_yield(),
            self.process.black_volatility(),
        )
    }
}

impl AsObservable for AnalyticPartialTimeBarrierOptionEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticPartialTimeBarrierOptionEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }

    fn results(&self) -> &dyn Results {
        self.base.results()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    fn calculate(&mut self) -> QlResult<()> {
        let arguments = self.base.arguments();
        let payoff = arguments.payoff.expect("validated");
        require!(payoff.strike() > 0.0, "strike must be positive");
        let spot = self.underlying()?;
        require!(spot > 0.0, "negative or null underlying given");

        let value = if payoff.option_type() == OptionType::Put {
            let spot_sq = spot * spot;
            let call_strike = spot_sq / payoff.strike();
            let call_payoff = PlainVanillaPayoff::new(OptionType::Call, call_strike);
            let tmp = PartialTimeBarrierArguments {
                barrier_type: Some(Self::symmetric_barrier_type(
                    arguments.barrier_type.expect("validated"),
                )),
                barrier_range: arguments.barrier_range,
                barrier: Some(spot_sq / arguments.barrier.expect("validated")),
                rebate: arguments.rebate,
                cover_event_date: arguments.cover_event_date,
                payoff: Some(call_payoff),
                exercise: arguments.exercise.clone(),
            };
            let call_process = self.swapped_process();
            payoff.strike() / spot
                * self.calculate_with(&tmp, tmp.payoff.as_ref().expect("set"), &call_process)?
        } else {
            self.calculate_with(arguments, &payoff, &self.process)?
        };

        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticPartialTimeBarrierOptionEngine`] to `option`.
pub fn set_analytic_partial_time_barrier_engine(
    option: &mut crate::instruments::PartialTimeBarrierOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine = shared_mut(AnalyticPartialTimeBarrierOptionEngine::new(process))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::PartialTimeBarrierOption;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn Quote>)
    }

    fn flat_rate(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::new(
            reference,
            quote_handle(quote),
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn flat_vol(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::with_quote(
            reference,
            None,
            quote_handle(quote),
            Actual360::new(),
        )) as Shared<dyn BlackVolTermStructure>)
    }

    struct Case {
        underlying: Real,
        strike: Real,
        days: i32,
        result: Real,
    }

    struct SymmetryCase {
        call_strike: Real,
        call_barrier: Real,
        call_type: BarrierType,
        put_strike: Real,
        put_barrier: Real,
        days: i32,
        put_type: BarrierType,
    }

    /// `partialtimebarrieroption.cpp` `testAnalyticEngine`.
    #[test]
    fn partial_time_barrier_end_b1_down_out_call_matches_quantlib() {
        let settings = shared(Settings::new());
        let today = Date::new(8, Month::August, 2025);
        settings.set_evaluation_date(today);
        let maturity = today + 360;
        let barrier = 100.0;

        let cases = [
            Case {
                underlying: 95.0,
                strike: 90.0,
                days: 1,
                result: 0.0393,
            },
            Case {
                underlying: 95.0,
                strike: 110.0,
                days: 1,
                result: 0.0000,
            },
            Case {
                underlying: 105.0,
                strike: 90.0,
                days: 1,
                result: 9.8751,
            },
            Case {
                underlying: 105.0,
                strike: 110.0,
                days: 1,
                result: 6.2303,
            },
            Case {
                underlying: 95.0,
                strike: 90.0,
                days: 90,
                result: 6.2747,
            },
            Case {
                underlying: 95.0,
                strike: 110.0,
                days: 90,
                result: 3.7352,
            },
            Case {
                underlying: 105.0,
                strike: 90.0,
                days: 90,
                result: 15.6324,
            },
            Case {
                underlying: 105.0,
                strike: 110.0,
                days: 90,
                result: 9.6812,
            },
            Case {
                underlying: 95.0,
                strike: 90.0,
                days: 180,
                result: 10.3345,
            },
            Case {
                underlying: 95.0,
                strike: 110.0,
                days: 180,
                result: 5.8712,
            },
            Case {
                underlying: 105.0,
                strike: 90.0,
                days: 180,
                result: 19.2896,
            },
            Case {
                underlying: 105.0,
                strike: 110.0,
                days: 180,
                result: 11.6055,
            },
            Case {
                underlying: 95.0,
                strike: 90.0,
                days: 270,
                result: 13.4342,
            },
            Case {
                underlying: 95.0,
                strike: 110.0,
                days: 270,
                result: 7.1270,
            },
            Case {
                underlying: 105.0,
                strike: 90.0,
                days: 270,
                result: 22.0753,
            },
            Case {
                underlying: 105.0,
                strike: 110.0,
                days: 270,
                result: 12.7342,
            },
            Case {
                underlying: 95.0,
                strike: 90.0,
                days: 359,
                result: 16.8576,
            },
            Case {
                underlying: 95.0,
                strike: 110.0,
                days: 359,
                result: 7.5763,
            },
            Case {
                underlying: 105.0,
                strike: 90.0,
                days: 359,
                result: 25.1488,
            },
            Case {
                underlying: 105.0,
                strike: 110.0,
                days: 359,
                result: 13.1376,
            },
        ];

        let spot = shared(SimpleQuote::new(0.0));
        let q_rate = shared(SimpleQuote::new(0.0));
        let r_rate = shared(SimpleQuote::new(0.1));
        let vol = shared(SimpleQuote::new(0.25));
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        for (i, case) in cases.iter().enumerate() {
            spot.set_value(case.underlying);
            let cover = today + case.days;
            let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(maturity));
            let mut option = PartialTimeBarrierOption::new(
                BarrierType::DownOut,
                PartialBarrierRange::EndB1,
                barrier,
                0.0,
                cover,
                PlainVanillaPayoff::new(OptionType::Call, case.strike),
                exercise,
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_partial_time_barrier_engine(&mut option, Shared::clone(&process));
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - case.result).abs() <= 1e-4,
                "case {i}: expected {}, got {calculated}",
                case.result
            );
        }
    }

    /// `partialtimebarrieroption.cpp` `testAnalyticEnginePutOption`.
    #[test]
    fn partial_time_barrier_end_b1_up_out_put_matches_quantlib() {
        let settings = shared(Settings::new());
        let today = Date::new(8, Month::August, 2025);
        settings.set_evaluation_date(today);
        let maturity = today + 360;
        let barrier = 100.0;

        let cases = [
            Case {
                underlying: 95.0,
                strike: 90.0,
                days: 1,
                result: 1.5551,
            },
            Case {
                underlying: 95.0,
                strike: 95.0,
                days: 1,
                result: 2.0589,
            },
            Case {
                underlying: 90.0,
                strike: 95.0,
                days: 1,
                result: 4.4512,
            },
            Case {
                underlying: 99.0,
                strike: 90.0,
                days: 1,
                result: 0.3404,
            },
            Case {
                underlying: 95.0,
                strike: 90.0,
                days: 90,
                result: 2.4181,
            },
            Case {
                underlying: 95.0,
                strike: 95.0,
                days: 90,
                result: 3.2257,
            },
            Case {
                underlying: 90.0,
                strike: 95.0,
                days: 90,
                result: 5.0624,
            },
            Case {
                underlying: 99.0,
                strike: 90.0,
                days: 90,
                result: 1.5992,
            },
            Case {
                underlying: 95.0,
                strike: 90.0,
                days: 180,
                result: 3.0021,
            },
            Case {
                underlying: 95.0,
                strike: 95.0,
                days: 180,
                result: 4.0617,
            },
            Case {
                underlying: 90.0,
                strike: 95.0,
                days: 180,
                result: 5.7960,
            },
            Case {
                underlying: 99.0,
                strike: 90.0,
                days: 180,
                result: 2.1903,
            },
            Case {
                underlying: 95.0,
                strike: 90.0,
                days: 270,
                result: 3.4194,
            },
            Case {
                underlying: 95.0,
                strike: 95.0,
                days: 270,
                result: 4.7362,
            },
            Case {
                underlying: 90.0,
                strike: 95.0,
                days: 270,
                result: 6.4370,
            },
            Case {
                underlying: 99.0,
                strike: 90.0,
                days: 270,
                result: 2.6025,
            },
            Case {
                underlying: 95.0,
                strike: 90.0,
                days: 359,
                result: 3.5965,
            },
            Case {
                underlying: 95.0,
                strike: 95.0,
                days: 359,
                result: 5.1865,
            },
            Case {
                underlying: 90.0,
                strike: 95.0,
                days: 359,
                result: 6.8782,
            },
            Case {
                underlying: 99.0,
                strike: 90.0,
                days: 359,
                result: 2.7759,
            },
        ];

        let spot = shared(SimpleQuote::new(0.0));
        let q_rate = shared(SimpleQuote::new(0.0));
        let r_rate = shared(SimpleQuote::new(0.1));
        let vol = shared(SimpleQuote::new(0.25));
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        for (i, case) in cases.iter().enumerate() {
            spot.set_value(case.underlying);
            let cover = today + case.days;
            let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(maturity));
            let mut option = PartialTimeBarrierOption::new(
                BarrierType::UpOut,
                PartialBarrierRange::EndB1,
                barrier,
                0.0,
                cover,
                PlainVanillaPayoff::new(OptionType::Put, case.strike),
                exercise,
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_partial_time_barrier_engine(&mut option, Shared::clone(&process));
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - case.result).abs() <= 1e-4,
                "case {i}: expected {}, got {calculated}",
                case.result
            );
        }
    }

    /// `partialtimebarrieroption.cpp` `testPutCallSymmetry`.
    #[test]
    fn partial_time_barrier_put_call_symmetry() {
        let settings = shared(Settings::new());
        let today = Date::new(8, Month::August, 2025);
        settings.set_evaluation_date(today);
        let maturity = today + 360;
        let spot_price = 100.0;
        let r = 0.01;

        let cases = [
            SymmetryCase {
                call_strike: 105.2631,
                call_barrier: 95.2380,
                call_type: BarrierType::DownOut,
                put_strike: 95.0,
                put_barrier: 105.0,
                days: 1,
                put_type: BarrierType::UpOut,
            },
            SymmetryCase {
                call_strike: 105.2631,
                call_barrier: 95.2380,
                call_type: BarrierType::DownOut,
                put_strike: 95.0,
                put_barrier: 105.0,
                days: 90,
                put_type: BarrierType::UpOut,
            },
            SymmetryCase {
                call_strike: 105.2631,
                call_barrier: 95.2380,
                call_type: BarrierType::DownOut,
                put_strike: 95.0,
                put_barrier: 105.0,
                days: 180,
                put_type: BarrierType::UpOut,
            },
            SymmetryCase {
                call_strike: 105.2631,
                call_barrier: 95.2380,
                call_type: BarrierType::DownOut,
                put_strike: 95.0,
                put_barrier: 105.0,
                days: 270,
                put_type: BarrierType::UpOut,
            },
            SymmetryCase {
                call_strike: 105.2631,
                call_barrier: 95.2380,
                call_type: BarrierType::DownOut,
                put_strike: 95.0,
                put_barrier: 105.0,
                days: 359,
                put_type: BarrierType::UpOut,
            },
            SymmetryCase {
                call_strike: 110.0,
                call_barrier: 120.0,
                call_type: BarrierType::UpOut,
                put_strike: 90.9090,
                put_barrier: 83.3333,
                days: 1,
                put_type: BarrierType::DownOut,
            },
            SymmetryCase {
                call_strike: 110.0,
                call_barrier: 120.0,
                call_type: BarrierType::UpOut,
                put_strike: 90.9090,
                put_barrier: 83.3333,
                days: 90,
                put_type: BarrierType::DownOut,
            },
            SymmetryCase {
                call_strike: 110.0,
                call_barrier: 120.0,
                call_type: BarrierType::UpOut,
                put_strike: 90.9090,
                put_barrier: 83.3333,
                days: 180,
                put_type: BarrierType::DownOut,
            },
            SymmetryCase {
                call_strike: 110.0,
                call_barrier: 120.0,
                call_type: BarrierType::UpOut,
                put_strike: 90.9090,
                put_barrier: 83.3333,
                days: 270,
                put_type: BarrierType::DownOut,
            },
            SymmetryCase {
                call_strike: 110.0,
                call_barrier: 120.0,
                call_type: BarrierType::UpOut,
                put_strike: 90.9090,
                put_barrier: 83.3333,
                days: 359,
                put_type: BarrierType::DownOut,
            },
        ];

        let spot = shared(SimpleQuote::new(spot_price));
        let q_rate = shared(SimpleQuote::new(0.0));
        let r_rate = shared(SimpleQuote::new(r));
        let vol = shared(SimpleQuote::new(0.25));

        let call_process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));
        let put_process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &r_rate),
            flat_rate(today, &q_rate),
            flat_vol(today, &vol),
        ));

        for (i, case) in cases.iter().enumerate() {
            let cover = today + case.days;
            let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(maturity));

            let mut put_option = PartialTimeBarrierOption::new(
                case.put_type,
                PartialBarrierRange::EndB1,
                case.put_barrier,
                0.0,
                cover,
                PlainVanillaPayoff::new(OptionType::Put, case.put_strike),
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_partial_time_barrier_engine(&mut put_option, Shared::clone(&put_process));

            let mut call_option = PartialTimeBarrierOption::new(
                case.call_type,
                PartialBarrierRange::EndB1,
                case.call_barrier,
                0.0,
                cover,
                PlainVanillaPayoff::new(OptionType::Call, case.call_strike),
                exercise,
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_partial_time_barrier_engine(
                &mut call_option,
                Shared::clone(&call_process),
            );

            let put_value = put_option.npv().unwrap();
            let call_value = call_option.npv().unwrap();
            let call_amount = case.put_strike / spot_price;
            let error = (put_value - call_amount * call_value).abs();
            assert!(
                error <= 1e-4,
                "symmetry case {i}: put={put_value}, call={call_value}, error={error}"
            );
        }
    }
}
