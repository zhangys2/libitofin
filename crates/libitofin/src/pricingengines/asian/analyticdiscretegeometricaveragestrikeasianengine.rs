//! Analytic discrete geometric-average strike Asian engine.
//!
//! Port of `ql/pricingengines/asian/analytic_discr_geom_av_strike.{hpp,cpp}`.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instrument::Instrument;
use crate::instruments::{
    AverageType, DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults, TypePayoff,
};
use crate::interestrate::Compounding;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::frequency::Frequency;
use crate::types::Real;

type EngineBase = GenericEngine<DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults>;

/// Pricing engine for European discrete geometric average-strike Asians.
pub struct AnalyticDiscreteGeometricAverageStrikeAsianEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl AnalyticDiscreteGeometricAverageStrikeAsianEngine {
    /// `AnalyticDiscreteGeometricAverageStrikeAsianEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(
            DiscreteAveragingAsianArguments::default(),
            DiscreteAveragingAsianResults::default(),
        );
        base.register_with(process.observable());
        Self { base, process }
    }
}

impl AsObservable for AnalyticDiscreteGeometricAverageStrikeAsianEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticDiscreteGeometricAverageStrikeAsianEngine {
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
        let average_type = args.average_type.expect("validated");
        let payoff = args.payoff.expect("validated");
        let exercise = args.exercise.as_ref().expect("validated");
        let running_accumulator = args.running_accumulator.expect("validated");
        let past_fixings = args.past_fixings.expect("validated");
        let fixing_dates = &args.fixing_dates;

        require!(
            average_type == AverageType::Geometric,
            "not a geometric average option"
        );
        if exercise.exercise_type() != ExerciseType::European {
            fail!("not an European option");
        }
        require!(
            running_accumulator > 0.0,
            "positive running product required: {running_accumulator} not allowed"
        );
        require!(past_fixings == 0, "past fixings currently not managed");

        let running_log = running_accumulator.ln();

        let vol_ts = self.process.black_volatility().current_link()?;
        let r_ts = self.process.risk_free_rate().current_link()?;
        let q_ts = self.process.dividend_yield().current_link()?;

        let rfdc = r_ts.require_day_counter()?;
        let divdc = q_ts.require_day_counter()?;
        let voldc = vol_ts.require_day_counter()?;

        let first_fixing = fixing_dates
            .first()
            .copied()
            .expect("validated fixing dates");
        let mut fixing_times = Vec::new();
        for fixing_date in fixing_dates {
            if *fixing_date >= first_fixing {
                fixing_times.push(voldc.year_fraction(first_fixing, *fixing_date));
            }
        }

        let remaining_fixings = fixing_times.len();
        let number_of_fixings = past_fixings + remaining_fixings;
        let n = number_of_fixings as Real;

        let past_weight = past_fixings as Real / n;
        let future_weight = 1.0 - past_weight;

        let time_sum: Real = fixing_times.iter().sum();
        let exercise_date = exercise.last_date();
        let residual_time = rfdc.year_fraction(first_fixing, exercise_date);

        let underlying = self.process.x0()?;
        require!(underlying > 0.0, "positive underlying value required");

        let volatility = vol_ts.black_vol_date(exercise_date, underlying, false)?;

        let dividend_rate = q_ts
            .zero_rate_date(
                exercise_date,
                divdc.clone(),
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let risk_free_rate = r_ts
            .zero_rate_date(
                exercise_date,
                rfdc.clone(),
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let nu = risk_free_rate - dividend_rate - 0.5 * volatility * volatility;

        let mut temp = 0.0;
        for i in (past_fixings + 1)..number_of_fixings {
            temp += fixing_times[i - past_fixings - 1] * (n - i as Real);
        }
        let variance = volatility * volatility / (n * n) * (time_sum + 2.0 * temp);
        let covariance_term = volatility * volatility / n * time_sum;
        let sigma_sum_2 =
            variance + volatility * volatility * residual_time - 2.0 * covariance_term;

        let m = if past_fixings == 0 { 1 } else { past_fixings };
        let running_log_average = running_log / m as Real;
        let mu_g = past_weight * running_log_average
            + future_weight * underlying.ln()
            + nu * time_sum / n;

        let cnd = CumulativeNormalDistribution::standard();
        let sqrt_sigma_sum_2 = sigma_sum_2.sqrt();
        let y1 = (underlying.ln()
            + (risk_free_rate - dividend_rate) * residual_time
            - mu_g
            - variance / 2.0
            + sigma_sum_2 / 2.0)
            / sqrt_sigma_sum_2;
        let y2 = y1 - sqrt_sigma_sum_2;

        let exp_neg_q_t = (-dividend_rate * residual_time).exp();
        let exp_mu_g = (mu_g + variance / 2.0 - risk_free_rate * residual_time).exp();

        let value = match payoff.option_type() {
            OptionType::Call => underlying * exp_neg_q_t * cnd.value(y1) - exp_mu_g * cnd.value(y2),
            OptionType::Put => {
                -underlying * exp_neg_q_t * cnd.value(-y1) + exp_mu_g * cnd.value(-y2)
            }
        };

        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticDiscreteGeometricAverageStrikeAsianEngine`] to `option`.
pub fn set_analytic_discrete_geometric_average_strike_asian_engine(
    option: &mut crate::instruments::DiscreteAveragingAsianOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine = shared_mut(AnalyticDiscreteGeometricAverageStrikeAsianEngine::new(process))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{DiscreteAveragingAsianOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::{Shared, shared};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn crate::quotes::Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn crate::quotes::Quote>)
    }

    fn flat_rate(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn YieldTermStructure> {
        Handle::new(
            shared(FlatForward::new(
                reference,
                quote_handle(quote),
                Actual360::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>,
        )
    }

    fn flat_vol(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(
            shared(BlackConstantVol::with_quote(
                reference,
                None,
                quote_handle(quote),
                Actual360::new(),
            )) as Shared<dyn BlackVolTermStructure>,
        )
    }

    /// `asianoptions.cpp` `testAnalyticDiscreteGeometricAverageStrike`.
    #[test]
    fn levy_discrete_geometric_average_strike_call() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(100.0));
        let q_rate = shared(SimpleQuote::new(0.03));
        let r_rate = shared(SimpleQuote::new(0.06));
        let vol = shared(SimpleQuote::new(0.20));

        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        let future_fixings = 10;
        let exercise_date = today + 360;
        let dt = (360.0 / future_fixings as Real).round() as i32;
        let mut fixing_dates = Vec::with_capacity(future_fixings);
        fixing_dates.push(today + dt);
        for _ in 1..future_fixings {
            let last = *fixing_dates.last().expect("non-empty");
            fixing_dates.push(last + dt);
        }

        let payoff = PlainVanillaPayoff::new(OptionType::Call, 100.0);
        let exercise = shared(EuropeanExercise::new(exercise_date));
        let mut option = DiscreteAveragingAsianOption::new(
            AverageType::Geometric,
            1.0,
            0,
            fixing_dates,
            payoff,
            exercise,
            settings,
        )
        .unwrap();

        set_analytic_discrete_geometric_average_strike_asian_engine(&mut option, process);

        let calculated = option.npv().unwrap();
        let expected = 4.97109;
        assert!(
            (calculated - expected).abs() <= 1.0e-5,
            "expected {expected}, got {calculated}"
        );
    }
}
