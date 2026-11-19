//! Finite-difference Black–Scholes engine for discrete arithmetic Asians.
//!
//! Port of `ql/pricingengines/asian/fdblackscholesasianengine.{hpp,cpp}` and
//! the arithmetic-average step condition it uses. European exercise and
//! arithmetic averaging only.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::instruments::{
    AverageType, DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults, Greeks,
    StrikedTypePayoff,
};
use crate::math::array::Array;
use crate::math::distributions::normal::InverseCumulativeNormal;
use crate::math::interpolations::Interpolation;
use crate::math::interpolations::cubic::MonotonicCubicNaturalSpline;
use crate::methods::finitedifferences::StepCondition;
use crate::methods::finitedifferences::meshers::{
    FdmMesher, FdmMesherComposite, fdm_black_scholes_mesher,
};
use crate::methods::finitedifferences::operators::FdmBlackScholesOp;
use crate::methods::finitedifferences::solvers::{Fdm2DimSolver, FdmSchemeDesc, FdmSolverDesc};
use crate::methods::finitedifferences::stepconditions::FdmStepConditionComposite;
use crate::methods::finitedifferences::utilities::fdm_log_inner_value;
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Real, Size, Time};

type EngineBase = GenericEngine<DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults>;

/// Finite-difference arithmetic Asian engine.
pub struct FdBlackScholesAsianEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    t_grid: Size,
    x_grid: Size,
    a_grid: Size,
    scheme_desc: FdmSchemeDesc,
}

impl FdBlackScholesAsianEngine {
    /// Douglas scheme with the QuantLib default grids `100 × 100 × 50`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self::with_grid(process, 100, 100, 50)
    }

    /// `FdBlackScholesAsianEngine(process, tGrid, xGrid, aGrid)`.
    pub fn with_grid(
        process: Shared<GeneralizedBlackScholesProcess>,
        t_grid: Size,
        x_grid: Size,
        a_grid: Size,
    ) -> Self {
        let base = EngineBase::new(
            DiscreteAveragingAsianArguments {
                average_type: None,
                running_accumulator: None,
                past_fixings: None,
                fixing_dates: Vec::new(),
                payoff: None,
                exercise: None,
            },
            DiscreteAveragingAsianResults::default(),
        );
        base.register_with(process.observable());
        Self {
            base,
            process,
            t_grid,
            x_grid,
            a_grid,
            scheme_desc: FdmSchemeDesc::douglas(),
        }
    }
}

impl AsObservable for FdBlackScholesAsianEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdBlackScholesAsianEngine {
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
        let args = self.base.arguments();
        args.validate()?;
        require!(
            args.average_type == Some(AverageType::Arithmetic),
            "Arithmetic averaging supported only"
        );
        let exercise = Shared::clone(args.exercise.as_ref().expect("validated"));
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "European exercise supported only"
        );
        let running = args.running_accumulator.expect("validated");
        let past_fixings = args.past_fixings.expect("validated");
        require!(
            running == 0.0 || past_fixings > 0,
            "Running average requires at least one past fixing"
        );
        let payoff = args.payoff.expect("validated");
        let fixing_dates = args.fixing_dates.clone();
        let _ = args;

        let maturity = self.process.time(&exercise.last_date())?;
        let strike = payoff.strike();
        let equity = fdm_black_scholes_mesher(
            self.x_grid,
            &self.process,
            maturity,
            strike,
            None,
            None,
            0.0001,
            1.5,
            None,
            &[],
            0.0,
        )?;
        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");
        let avg = if running == 0.0 {
            spot
        } else {
            running / past_fixings as Real
        };
        let norm_inv_eps = InverseCumulativeNormal::standard_value(1.0 - 0.0001)?;
        let sigma_sqrt_t = self
            .process
            .black_volatility()
            .current_link()?
            .black_vol(maturity, strike, false)?
            * maturity.sqrt();
        let radius = sigma_sqrt_t * norm_inv_eps;
        let x_min = (avg.ln() - 0.25 * radius).min(spot.ln() - 1.5 * radius);
        let x_max = (avg.ln() + 0.25 * radius).max(spot.ln() + 1.5 * radius);
        let average = fdm_black_scholes_mesher(
            self.a_grid,
            &self.process,
            maturity,
            strike,
            Some(x_min),
            Some(x_max),
            0.0001,
            1.5,
            None,
            &[],
            0.0,
        )?;
        let mesher =
            shared(FdmMesherComposite::new(vec![equity, average])) as Shared<dyn FdmMesher>;
        let mut average_times = Vec::with_capacity(fixing_dates.len());
        for date in &fixing_dates {
            let t = self.process.time(date)?;
            require!(t >= 0.0, "Fixing dates must not contain past date");
            average_times.push(t);
        }
        let condition = shared(FdmArithmeticAverageCondition::new(
            average_times.clone(),
            past_fixings,
            Shared::clone(&mesher),
            0,
        )?);
        let conditions = shared(FdmStepConditionComposite::new(
            &[average_times],
            vec![condition as Shared<dyn StepCondition>],
        ));
        let calculator = shared(fdm_log_inner_value(
            shared(payoff) as Shared<dyn Payoff>,
            Shared::clone(&mesher),
            1,
        ))
            as Shared<dyn crate::methods::finitedifferences::utilities::FdmInnerValueCalculator>;
        let solver_desc = FdmSolverDesc {
            mesher: Shared::clone(&mesher),
            bc_set: Vec::new(),
            condition: conditions,
            calculator,
            maturity,
            time_steps: self.t_grid,
            damping_steps: 0,
        };
        let op = shared_mut(FdmBlackScholesOp::new(
            Shared::clone(&mesher),
            &self.process,
            strike,
            0,
        )?)
            as SharedMut<dyn crate::methods::finitedifferences::operators::FdmLinearOpComposite>;
        let solver = Fdm2DimSolver::new(solver_desc, self.scheme_desc, op);
        let value = solver.interpolate_at(spot.ln(), avg.ln())?;
        let eps = spot * 0.01;
        let up = solver.interpolate_at((spot + eps).ln(), avg.ln())?;
        let down = solver.interpolate_at((spot - eps).ln(), avg.ln())?;
        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        results.greeks = Greeks {
            delta: Some((up - down) / (2.0 * eps)),
            gamma: Some((up + down - 2.0 * value) / (eps * eps)),
            ..Greeks::default()
        };
        Ok(())
    }
}

struct FdmArithmeticAverageCondition {
    x: Vec<Real>,
    a: Vec<Real>,
    average_times: Vec<Time>,
    past_fixings: Size,
    mesher: Shared<dyn FdmMesher>,
    equity_direction: Size,
}

impl FdmArithmeticAverageCondition {
    fn new(
        average_times: Vec<Time>,
        past_fixings: Size,
        mesher: Shared<dyn FdmMesher>,
        equity_direction: Size,
    ) -> QlResult<Self> {
        let layout = mesher.layout();
        require!(layout.dim().len() == 2, "2D allowed only");
        require!(
            equity_direction == 0 || equity_direction == 1,
            "equityDirection has to be 0 or 1"
        );
        let x_spacing = layout.spacing()[equity_direction];
        let x_locations = mesher.locations(equity_direction);
        let x_count = layout.dim()[equity_direction];
        let mut x = Vec::with_capacity(x_count);
        for i in 0..x_count {
            x.push(x_locations[i * x_spacing].exp());
        }
        let average_direction = if equity_direction == 0 { 1 } else { 0 };
        let a_spacing = layout.spacing()[average_direction];
        let a_locations = mesher.locations(average_direction);
        let a_count = layout.dim()[average_direction];
        let mut a = Vec::with_capacity(a_count);
        for i in 0..a_count {
            a.push(a_locations[i * a_spacing].exp());
        }
        Ok(Self {
            x,
            a,
            average_times,
            past_fixings,
            mesher,
            equity_direction,
        })
    }
}

impl StepCondition for FdmArithmeticAverageCondition {
    fn apply_to(&self, values: &mut Array, t: Time) {
        let hits: Vec<usize> = self
            .average_times
            .iter()
            .enumerate()
            .filter(|(_, time)| **time == t)
            .map(|(i, _)| i)
            .collect();
        if hits.is_empty() {
            return;
        }
        let n_times = hits.len();
        let i_t = hits[0] + 1 + self.past_fixings;
        let average_direction = if self.equity_direction == 0 { 1 } else { 0 };
        let x_spacing = self.mesher.layout().spacing()[self.equity_direction];
        let a_spacing = self.mesher.layout().spacing()[average_direction];
        let copy = values.clone();
        let weight_old = (i_t - n_times) as Real / i_t as Real;
        let weight_new = n_times as Real / i_t as Real;
        for (i, &equity) in self.x.iter().enumerate() {
            let mut column = Vec::with_capacity(self.a.len());
            for j in 0..self.a.len() {
                column.push(copy[i * x_spacing + j * a_spacing]);
            }
            let Ok(interp) = MonotonicCubicNaturalSpline::new(self.a.clone(), column) else {
                continue;
            };
            let interp = interp.with_extrapolation(true);
            for j in 0..self.a.len() {
                let query = weight_old * self.a[j] + weight_new * equity;
                if let Ok(updated) = interp.value(query) {
                    values[i * x_spacing + j * a_spacing] = updated;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{DiscreteAveragingAsianOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType::Put;
    use crate::pricingengine::PricingEngine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;

    /// Levy (1997) rows in `asianoptions.cpp` `testMCDiscreteArithmeticAveragePrice`
    /// that the finite-difference engine prices (`fixings < 100`) at tolerance `2e-2`.
    #[test]
    fn fd_matches_levy_arithmetic_table() {
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let spot = shared(SimpleQuote::new(90.0));
        let q_rate = shared(SimpleQuote::new(0.06));
        let r_rate = shared(SimpleQuote::new(0.025));
        let vol = shared(SimpleQuote::new(0.13));
        let quote = |q: &Shared<SimpleQuote>| Handle::new(Shared::clone(q) as Shared<dyn Quote>);
        let flat = |q: &Shared<SimpleQuote>| {
            Handle::new(shared(FlatForward::new(
                today,
                quote(q),
                Actual360::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        let process = shared(BlackScholesMertonProcess::new(
            quote(&spot),
            flat(&q_rate),
            flat(&r_rate),
            Handle::new(shared(BlackConstantVol::with_quote(
                today,
                None,
                quote(&vol),
                Actual360::new(),
            )) as Shared<dyn BlackVolTermStructure>),
        ));
        let engine = shared_mut(FdBlackScholesAsianEngine::with_grid(
            Shared::clone(&process),
            100,
            100,
            100,
        ));
        // (first, fixings, expected) for underlying 90, strike 87, q 0.06, r 0.025, vol 0.13, length 11/12.
        let cases = [
            (0.0, 2usize, 1.3942835683),
            (0.0, 4, 1.5852442983),
            (0.0, 8, 1.66970673),
            (0.0, 12, 1.6980019214),
            (0.0, 26, 1.7255070456),
            (0.0, 52, 1.7401553533),
            (1.0 / 12.0, 2, 1.8496053697),
            (1.0 / 12.0, 4, 2.0111495205),
            (1.0 / 12.0, 8, 2.0852138818),
            (1.0 / 12.0, 12, 2.1105094397),
            (1.0 / 12.0, 26, 2.1346526695),
            (1.0 / 12.0, 52, 2.147489651),
            (3.0 / 12.0, 2, 2.63315092584),
            (3.0 / 12.0, 4, 2.76723962361),
            (3.0 / 12.0, 8, 2.83124836881),
            (3.0 / 12.0, 12, 2.84290301412),
            (3.0 / 12.0, 26, 2.88179560417),
            (3.0 / 12.0, 52, 2.88447044543),
        ];
        let length = 11.0 / 12.0;
        for (first, fixings, expected) in cases {
            let dt = length / (fixings - 1) as Real;
            let mut dates = Vec::with_capacity(fixings);
            for i in 0..fixings {
                let t = if i == 0 {
                    first
                } else {
                    i as Real * dt + first
                };
                dates.push(today + (t * 360.0).round() as i32);
            }
            let exercise = shared(EuropeanExercise::new(*dates.last().unwrap()));
            let mut option = DiscreteAveragingAsianOption::new(
                AverageType::Arithmetic,
                0.0,
                0,
                dates,
                PlainVanillaPayoff::new(Put, 87.0),
                exercise,
                Shared::clone(&settings),
            )
            .unwrap();
            option
                .base_mut()
                .set_pricing_engine(SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>);
            let calculated = option.npv().unwrap();
            let err = (calculated - expected).abs();
            assert!(
                err <= 2.0e-2,
                "first={first} n={fixings}: {calculated} vs {expected} (err={err})"
            );
        }
    }
}
