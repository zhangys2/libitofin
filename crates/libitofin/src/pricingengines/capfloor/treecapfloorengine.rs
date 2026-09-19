//! Numerical lattice engine for caps and floors.
//!
//! Port of `ql/pricingengines/capfloor/treecapfloorengine.{hpp,cpp}` with the
//! folded-in `LatticeShortRateModelEngine`. Bound to [`HullWhite`] (same as
//! [`TreeSwaptionEngine`](crate::pricingengines::TreeSwaptionEngine)): grows the
//! model tree over [`DiscretizedCapFloor`] mandatory times and rolls back to
//! the first start date.
//!
//! The time-steps path requires `start_dates[0] >= reference_date` (negative
//! mandatory times are rejected by [`TimeGrid::with_mandatory_times`], same as
//! QL). Past-start optionlets need the deferred fixed-`TimeGrid` ctor.
//!
//! Deferred: fixed-`TimeGrid` ctor, non-TS-consistent `termStructure_` fallback,
//! MC / Gaussian1d cap engines, `CapHelper::addTimesTo`.

use crate::discretizedasset::DiscretizedAsset;
use crate::errors::QlResult;
use crate::instrument::InstrumentResults;
use crate::instruments::CapFloorArguments;
use crate::math::timegrid::TimeGrid;
use crate::methods::lattices::lattice::Lattice;
use crate::models::model::CalibratedModelHolder;
use crate::models::shortrate::hullwhite::HullWhite;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::require;
use crate::shared::{Shared, SharedMut, shared};
use crate::types::{Size, Time};

use super::discretizedcapfloor::DiscretizedCapFloor;

/// Lattice cap/floor engine under [`HullWhite`] (`treecapfloorengine.hpp`).
pub struct TreeCapFloorEngine {
    base: GenericEngine<CapFloorArguments, InstrumentResults>,
    model: SharedMut<HullWhite>,
    time_steps: Size,
}

impl TreeCapFloorEngine {
    /// Hull-White model with a fixed step count (`…cpp:26` + lattice base).
    pub fn new(model: SharedMut<HullWhite>, time_steps: Size) -> QlResult<Self> {
        require!(
            time_steps > 0,
            "timeSteps must be positive, {time_steps} not allowed"
        );
        let base = GenericEngine::new(CapFloorArguments::default(), InstrumentResults::default());
        base.register_with(model.borrow().calibrated_model().observable());
        Ok(Self {
            base,
            model,
            time_steps,
        })
    }
}

impl AsObservable for TreeCapFloorEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for TreeCapFloorEngine {
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
        let model = self.model.borrow();
        let (reference_date, day_counter) = {
            let curve = model.term_structure().current_link()?;
            (curve.reference_date()?, curve.require_day_counter()?)
        };

        let args = self.base.arguments().clone();
        let Some(&first_start) = args.start_dates.first() else {
            crate::fail!("cap/floor has no optionlets");
        };
        let Some(&last_end) = args.end_dates.last() else {
            crate::fail!("cap/floor has no optionlets");
        };
        let first_time: Time = day_counter.year_fraction(reference_date, first_start);
        let last_time: Time = day_counter.year_fraction(reference_date, last_end);
        require!(
            first_time >= 0.0,
            "TreeCapFloorEngine time-steps path needs start_dates[0] >= reference date \
             (got first start {first_time}); past-start optionlets need the deferred \
             fixed-TimeGrid ctor"
        );

        let mut capfloor = DiscretizedCapFloor::new(args, reference_date, &day_counter);
        let times = capfloor.mandatory_times();
        let grid = TimeGrid::with_mandatory_times(&times, self.time_steps)?;
        let lattice: Shared<dyn Lattice> = shared(model.tree(grid)?);
        drop(model);

        capfloor.initialize(lattice, last_time)?;
        capfloor.rollback(first_time)?;
        self.base.results_mut().value = Some(capfloor.present_value()?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cashflows::{IborCoupon, IborLeg};
    use crate::handle::Handle;
    use crate::indexes::IborIndex;
    use crate::indexes::ibor::Euribor;
    use crate::instrument::Instrument;
    use crate::instruments::CapFloor;
    use crate::interestrate::Compounding;
    use crate::pricingengines::AnalyticCapFloorEngine;
    use crate::settings::Settings;
    use crate::shared::{shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::target::Target;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::time::schedule::MakeSchedule;
    use crate::time::timeunit::TimeUnit;
    use crate::types::{Rate, Real};

    const A: Real = 0.05;
    const SIGMA: Real = 0.01;
    const NOMINAL: Real = 100.0;
    const K: Rate = 0.03;

    fn settings() -> Shared<Settings<Date>> {
        let s = shared(Settings::<Date>::new());
        s.set_evaluation_date(Date::new(15, Month::January, 2026));
        s
    }

    fn curve() -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            Date::new(15, Month::January, 2026),
            0.03,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn hw() -> SharedMut<HullWhite> {
        HullWhite::new(curve(), A, SIGMA).unwrap()
    }

    fn coupons(settings: &Shared<Settings<Date>>) -> Vec<Shared<IborCoupon>> {
        let index: Shared<IborIndex> =
            shared(Euribor::six_months(curve(), Shared::clone(settings)));
        let from = Date::new(15, Month::January, 2027);
        let to = Target::new().advance(
            from,
            2,
            TimeUnit::Years,
            BusinessDayConvention::ModifiedFollowing,
            false,
        );
        let schedule = MakeSchedule::new()
            .from(from)
            .to(to)
            .with_frequency(Frequency::Semiannual)
            .with_calendar(Target::new())
            .with_convention(BusinessDayConvention::ModifiedFollowing)
            .with_termination_date_convention(BusinessDayConvention::ModifiedFollowing)
            .forwards()
            .end_of_month(false)
            .build();
        IborLeg::new(schedule, index)
            .with_notional(NOMINAL)
            .coupons()
            .unwrap()
    }

    fn tree_npv(is_cap: bool, steps: Size) -> Real {
        let settings = settings();
        let mut cf = if is_cap {
            CapFloor::cap(coupons(&settings), vec![K], Shared::clone(&settings))
        } else {
            CapFloor::floor(coupons(&settings), vec![K], Shared::clone(&settings))
        }
        .unwrap();
        cf.base_mut()
            .set_pricing_engine(shared_mut(TreeCapFloorEngine::new(hw(), steps).unwrap())
                as SharedMut<dyn PricingEngine>);
        cf.npv().unwrap()
    }

    fn analytic_npv(is_cap: bool) -> Real {
        let settings = settings();
        let mut cf = if is_cap {
            CapFloor::cap(coupons(&settings), vec![K], Shared::clone(&settings))
        } else {
            CapFloor::floor(coupons(&settings), vec![K], Shared::clone(&settings))
        }
        .unwrap();
        cf.base_mut()
            .set_pricing_engine(shared_mut(AnalyticCapFloorEngine::new(
                hw(),
                Shared::clone(&settings),
            )) as SharedMut<dyn PricingEngine>);
        cf.npv().unwrap()
    }

    #[test]
    fn tree_converges_toward_analytic() {
        // Fixture analytic pins (flat 3%, HW(0.05,0.01), 2Y 6M Euribor, K=3%).
        const CAP_ANALYTIC: Real = 0.91417378;
        const FLOOR_ANALYTIC: Real = 0.94978244;

        let cap_ref = analytic_npv(true);
        assert!(
            (cap_ref - CAP_ANALYTIC).abs() < 1.0e-6,
            "cap analytic fixture drifted: {cap_ref} vs {CAP_ANALYTIC}"
        );
        let floor_ref = analytic_npv(false);
        assert!(
            (floor_ref - FLOOR_ANALYTIC).abs() < 1.0e-6,
            "floor analytic fixture drifted: {floor_ref} vs {FLOOR_ANALYTIC}"
        );

        let e_coarse = (tree_npv(true, 40) - cap_ref).abs() / cap_ref;
        let e_fine = (tree_npv(true, 400) - cap_ref).abs() / cap_ref;
        assert!(
            e_fine < 5.0e-3,
            "cap tree(400) rel err {e_fine} vs analytic {cap_ref}"
        );
        assert!(
            e_fine < e_coarse,
            "cap error must shrink 40->400: coarse {e_coarse} fine {e_fine}"
        );

        let floor_e = (tree_npv(false, 400) - floor_ref).abs() / floor_ref;
        assert!(
            floor_e < 5.0e-3,
            "floor tree(400) rel err {floor_e} vs analytic {floor_ref}"
        );
    }

    /// Type-dispatch smoke: collar = long cap + short floor on one tree.
    #[test]
    fn collar_type_dispatch_matches_cap_minus_floor() {
        let settings = settings();
        let c = coupons(&settings);
        let eng = || {
            shared_mut(TreeCapFloorEngine::new(hw(), 120).unwrap()) as SharedMut<dyn PricingEngine>
        };
        let mut collar =
            CapFloor::collar(c.clone(), vec![0.04], vec![0.02], Shared::clone(&settings)).unwrap();
        collar.base_mut().set_pricing_engine(eng());
        let mut cap = CapFloor::cap(c.clone(), vec![0.04], Shared::clone(&settings)).unwrap();
        cap.base_mut().set_pricing_engine(eng());
        let mut floor = CapFloor::floor(c, vec![0.02], settings).unwrap();
        floor.base_mut().set_pricing_engine(eng());
        assert!(((cap.npv().unwrap() - floor.npv().unwrap()) - collar.npv().unwrap()).abs() < 1e-8);
    }

    #[test]
    fn rejects_non_positive_time_steps() {
        let err = match TreeCapFloorEngine::new(hw(), 0) {
            Err(e) => e,
            Ok(_) => panic!("expected rejection"),
        };
        assert!(err.message().contains("timeSteps must be positive"));
    }
}
