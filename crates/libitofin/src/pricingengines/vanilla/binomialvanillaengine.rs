//! Binomial-tree engine for vanilla options.
//!
//! Port of QuantLib's `ql/pricingengines/vanilla/binomialengine.hpp` for the
//! Cox-Ross-Rubinstein tree: it extracts flat `r`, `q`, `sigma` from the
//! process at maturity, builds a CRR [`BlackScholesLattice`], and rolls a
//! discretized vanilla option back to today. European and American (and
//! Bermudan) exercise are supported; the value converges to the analytic
//! Black-Scholes price.

use crate::discretizedasset::{DiscretizedAsset, DiscretizedAssetBase};
use crate::errors::QlResult;
use crate::exercise::{Exercise, ExerciseType};
use crate::fail;
use crate::instruments::{
    Greeks, MoreGreeks, OneAssetOptionEngine, OneAssetOptionResults, OptionArguments,
    StrikedTypePayoff,
};
use crate::math::array::Array;
use crate::math::timegrid::TimeGrid;
use crate::methods::lattices::binomialtree::CoxRossRubinstein;
use crate::methods::lattices::lattice::Lattice;
use crate::methods::lattices::treelattice::{TreeLattice1D, TreeLatticeImpl};
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Real, Size, Time};

/// Time tolerance for the American exercise window.
const EPS: Time = 1.0e-10;

/// A constant-coefficient Black-Scholes binomial lattice over a
/// [`CoxRossRubinstein`] tree (`bsmlattice.hpp` `BlackScholesLattice`).
struct BlackScholesLattice {
    tree: CoxRossRubinstein,
    grid: TimeGrid,
    risk_free_rate: Real,
}

impl TreeLatticeImpl for BlackScholesLattice {
    type Tree = CoxRossRubinstein;

    fn tree(&self) -> &CoxRossRubinstein {
        &self.tree
    }

    fn discount(&self, i: Size, _index: Size) -> Real {
        (-self.risk_free_rate * self.grid.dt(i)).exp()
    }
}

/// The discretized vanilla option rolled back on the lattice
/// (`discretizedvanillaoption.cpp`).
struct DiscretizedVanillaOption {
    base: DiscretizedAssetBase,
    payoff: Shared<dyn StrikedTypePayoff>,
    exercise_type: ExerciseType,
    stopping_times: Vec<Time>,
}

impl DiscretizedVanillaOption {
    fn new(
        payoff: Shared<dyn StrikedTypePayoff>,
        exercise: &Shared<dyn Exercise>,
        process: &GeneralizedBlackScholesProcess,
        grid: &TimeGrid,
    ) -> QlResult<DiscretizedVanillaOption> {
        let mut stopping_times = Vec::with_capacity(exercise.dates().len());
        for &date in exercise.dates() {
            let time = process.time(&date)?;
            let index = grid.closest_index(time);
            stopping_times.push(grid.times()[index]);
        }
        Ok(DiscretizedVanillaOption {
            base: DiscretizedAssetBase::default(),
            payoff,
            exercise_type: exercise.exercise_type(),
            stopping_times,
        })
    }

    fn apply_specific_condition(&mut self) -> QlResult<()> {
        let lattice = self.require_method()?;
        let grid = lattice.grid(self.time())?;
        let payoff = Shared::clone(&self.payoff);
        let values = self.values_mut();
        for j in 0..values.size() {
            values[j] = values[j].max(payoff.value(grid[j]));
        }
        Ok(())
    }
}

impl DiscretizedAsset for DiscretizedVanillaOption {
    fn base(&self) -> &DiscretizedAssetBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut DiscretizedAssetBase {
        &mut self.base
    }

    fn as_asset_mut(&mut self) -> &mut dyn DiscretizedAsset {
        self
    }

    fn reset(&mut self, size: Size) -> QlResult<()> {
        *self.values_mut() = Array::filled(size, 0.0);
        self.adjust_values()
    }

    fn mandatory_times(&self) -> Vec<Time> {
        self.stopping_times.clone()
    }

    fn post_adjust_values_impl(&mut self) -> QlResult<()> {
        let now = self.time();
        match self.exercise_type {
            ExerciseType::American => {
                if now >= self.stopping_times[0] - EPS && now <= self.stopping_times[1] + EPS {
                    self.apply_specific_condition()?;
                }
            }
            ExerciseType::European => {
                if self.is_on_time(self.stopping_times[0]) {
                    self.apply_specific_condition()?;
                }
            }
            ExerciseType::Bermudan => {
                for i in 0..self.stopping_times.len() {
                    if self.is_on_time(self.stopping_times[i]) {
                        self.apply_specific_condition()?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// A Cox-Ross-Rubinstein binomial engine for vanilla options.
pub struct BinomialVanillaEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
    time_steps: Size,
}

impl BinomialVanillaEngine {
    /// Builds the engine over `process` with `time_steps` binomial steps.
    ///
    /// # Errors
    ///
    /// Fails when `time_steps` is below 2.
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        time_steps: Size,
    ) -> QlResult<BinomialVanillaEngine> {
        require!(
            time_steps >= 2,
            "at least 2 time steps required, {time_steps} provided"
        );
        let base =
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(process.observable());
        Ok(BinomialVanillaEngine {
            base,
            process,
            time_steps,
        })
    }
}

impl AsObservable for BinomialVanillaEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for BinomialVanillaEngine {
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
        let Some(exercise) = &arguments.exercise else {
            fail!("no exercise given");
        };
        let exercise = Shared::clone(exercise);
        let Some(payoff) = &arguments.payoff else {
            fail!("no payoff given");
        };
        let payoff = Shared::clone(payoff);

        let maturity_date = exercise.last_date();
        let maturity = self.process.time(&maturity_date)?;
        if maturity <= 0.0 {
            fail!("the binomial engine needs a positive maturity");
        }
        let spot = self.process.x0()?;

        let risk_free = self.process.risk_free_rate().current_link()?;
        let dividend = self.process.dividend_yield().current_link()?;
        let vol = self.process.black_volatility().current_link()?;
        let r = -risk_free.discount(maturity, false)?.ln() / maturity;
        let q = -dividend.discount(maturity, false)?.ln() / maturity;
        let v = vol.black_vol(maturity, spot, true)?;

        let tree = CoxRossRubinstein::new(spot, r, q, v, maturity, self.time_steps)?;
        let grid = TimeGrid::new(maturity, self.time_steps)?;
        let bsl = BlackScholesLattice {
            tree,
            grid: grid.clone(),
            risk_free_rate: r,
        };
        let lattice: Shared<dyn Lattice> = shared(TreeLattice1D::new(bsl, grid.clone())?);

        let mut option = DiscretizedVanillaOption::new(payoff, &exercise, &self.process, &grid)?;
        option.initialize(Shared::clone(&lattice), maturity)?;
        option.rollback(0.0)?;
        let value = option.present_value()?;

        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        results.greeks = Greeks::default();
        results.more_greeks = MoreGreeks::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{AmericanExercise, EuropeanExercise};
    use crate::instrument::Instrument;
    use crate::instruments::{OneAssetOption, PlainVanillaPayoff};
    use crate::option::OptionType;
    use crate::pricingengines::vanilla::test_market::{market, time_to_days, today};
    use crate::shared::{SharedMut, shared_mut};

    fn european_binomial(option_type: OptionType, strike: Real, steps: Size) -> Real {
        let market = market();
        market.set(100.0, 0.0, 0.05, 0.20);
        let expiry = today() + time_to_days(1.0);
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(option_type, strike));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));
        let mut option = OneAssetOption::new(payoff, exercise, Shared::clone(&market.settings));
        let engine =
            shared_mut(BinomialVanillaEngine::new(Shared::clone(&market.process), steps).unwrap());
        option
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        option.npv().unwrap()
    }

    fn analytic_european(option_type: OptionType, strike: Real) -> Real {
        let market = market();
        market.set(100.0, 0.0, 0.05, 0.20);
        let expiry = today() + time_to_days(1.0);
        let mut option = market.option(option_type, strike, expiry);
        option.npv().unwrap()
    }

    #[test]
    fn binomial_european_converges_to_black_scholes() {
        let analytic = analytic_european(OptionType::Call, 100.0);
        let coarse = (european_binomial(OptionType::Call, 100.0, 50) - analytic).abs();
        let fine = (european_binomial(OptionType::Call, 100.0, 1000) - analytic).abs();
        assert!(
            fine < 5.0e-2,
            "binomial call {} vs analytic {analytic} (err {fine})",
            european_binomial(OptionType::Call, 100.0, 1000)
        );
        assert!(fine < coarse, "refining steps should reduce the error");
    }

    #[test]
    fn american_put_dominates_the_european_put() {
        let market = market();
        market.set(100.0, 0.0, 0.05, 0.20);
        let expiry = today() + time_to_days(1.0);
        let euro = european_binomial(OptionType::Put, 110.0, 500);

        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Put, 110.0));
        let exercise: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today(), expiry, false).unwrap());
        let mut american = OneAssetOption::new(payoff, exercise, Shared::clone(&market.settings));
        let engine =
            shared_mut(BinomialVanillaEngine::new(Shared::clone(&market.process), 500).unwrap());
        american
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        let amer = american.npv().unwrap();

        assert!(
            amer > euro + 1e-4,
            "american put {amer} should exceed european put {euro}"
        );
    }
}
