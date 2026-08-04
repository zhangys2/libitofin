//! Tree engine for callable fixed-rate bonds.
//!
//! Port of QuantLib's
//! `ql/experimental/callablebonds/treecallablebondengine.{hpp,cpp}` for the
//! Hull-White short-rate model: it builds a trinomial lattice fitted to the
//! model's term structure, rolls the [`DiscretizedCallableFixedRateBond`] back
//! to today, and reads the present value.

use super::discretizedcallablebond::DiscretizedCallableFixedRateBond;
use crate::discretizedasset::DiscretizedAsset;
use crate::errors::QlResult;
use crate::instruments::{BondResults, CallableBondArguments};
use crate::math::timegrid::TimeGrid;
use crate::methods::lattices::lattice::Lattice;
use crate::models::model::CalibratedModelHolder;
use crate::models::shortrate::HullWhite;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::{Shared, SharedMut, shared};
use crate::time::date::Date;
use crate::types::Size;

type CallableBondEngineBase = GenericEngine<CallableBondArguments, BondResults>;

/// A Hull-White lattice engine for callable fixed-rate bonds.
pub struct TreeCallableFixedRateBondEngine {
    base: CallableBondEngineBase,
    model: SharedMut<HullWhite>,
    time_steps: Size,
    settings: Shared<Settings<Date>>,
}

impl TreeCallableFixedRateBondEngine {
    /// Builds the engine over `model` with `time_steps` lattice steps.
    ///
    /// # Errors
    ///
    /// Fails when `time_steps` is zero.
    pub fn new(
        model: SharedMut<HullWhite>,
        time_steps: Size,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<TreeCallableFixedRateBondEngine> {
        require!(
            time_steps > 0,
            "timeSteps must be positive, {time_steps} not allowed"
        );
        let base =
            CallableBondEngineBase::new(CallableBondArguments::default(), BondResults::default());
        base.register_with(model.borrow().calibrated_model().observable());
        Ok(TreeCallableFixedRateBondEngine {
            base,
            model,
            time_steps,
            settings,
        })
    }
}

impl AsObservable for TreeCallableFixedRateBondEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for TreeCallableFixedRateBondEngine {
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
        let _ = &self.settings;
        let (curve, reference_date, day_counter, settlement_date, redemption_date) = {
            let model = self.model.borrow();
            let curve = model.term_structure().current_link()?;
            let reference_date = curve.reference_date()?;
            let day_counter = curve.require_day_counter()?;
            let args = self.base.arguments();
            let settlement_date = args.settlement_date.expect("validated settlement date");
            let redemption_date = args.redemption_date.expect("validated redemption date");
            (
                curve,
                reference_date,
                day_counter,
                settlement_date,
                redemption_date,
            )
        };

        let mut bond = DiscretizedCallableFixedRateBond::new(self.base.arguments(), &*curve)?;

        let times = bond.mandatory_times();
        let grid = TimeGrid::with_mandatory_times(&times, self.time_steps)?;
        let lattice: Shared<dyn Lattice> = {
            let model = self.model.borrow();
            shared(model.tree(grid)?)
        };

        let redemption_time = day_counter.year_fraction(reference_date, redemption_date);
        bond.initialize(Shared::clone(&lattice), redemption_time)?;
        bond.rollback(0.0)?;
        let value = bond.present_value()?;
        let settlement_value = value / curve.discount_date(settlement_date, true)?;

        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        results.settlement_value = Some(settlement_value);
        Ok(())
    }
}
