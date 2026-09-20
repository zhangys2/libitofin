//! Bachelier (normal) cap/floor engine (`bacheliercapfloorengine.{hpp,cpp}`).
//! Prices [`CapFloor`](crate::instruments::CapFloor) optionlets with
//! [`bachelier_black_formula`] on a Normal surface; reports `value`/`vega`/
//! `optionletsPrice`/`optionletsDelta`/`optionletsDiscountFactor`/
//! `optionletsAtmForward`. Flat-vol forces Normal (QL defaults ShiftedLognormal);
//! explicit [`Settings`] (D5).

use crate::errors::QlResult;
use crate::handle::Handle;
use crate::instrument::InstrumentResults;
use crate::instruments::{CapFloorArguments, CapFloorType};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::blackformula::{
    bachelier_black_formula, bachelier_black_formula_asset_itm_probability,
    bachelier_black_formula_std_dev_derivative,
};
use crate::quotes::Quote;
use crate::settings::Settings;
use crate::shared::{Shared, shared};
use crate::termstructures::volatility::{
    ConstantOptionletVolatility, OptionletVolatilityStructure, VolatilityType,
};
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::calendars::nullcalendar::NullCalendar;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::types::Real;
use crate::{fail, require};
use std::any::Any;

/// Normal-model engine for caps, floors and collars.
pub struct BachelierCapFloorEngine {
    base: GenericEngine<CapFloorArguments, InstrumentResults>,
    discount_curve: Handle<dyn YieldTermStructure>,
    vol: Handle<dyn OptionletVolatilityStructure>,
}

impl BachelierCapFloorEngine {
    /// Discount curve + Normal optionlet surface (`…cpp:62-73`).
    pub fn new(
        discount_curve: Handle<dyn YieldTermStructure>,
        vol: Handle<dyn OptionletVolatilityStructure>,
    ) -> QlResult<Self> {
        let surface = vol.current_link()?;
        require!(
            surface.volatility_type() == VolatilityType::Normal,
            "BachelierCapFloorEngine needs a normal optionlet surface"
        );
        drop(surface);

        let base = GenericEngine::new(CapFloorArguments::default(), InstrumentResults::default());
        discount_curve.register_observer(&base.observer());
        vol.register_observer(&base.observer());
        Ok(Self {
            base,
            discount_curve,
            vol,
        })
    }

    /// Flat Normal volatility on a moving [`ConstantOptionletVolatility`].
    pub fn with_flat_vol(
        discount_curve: Handle<dyn YieldTermStructure>,
        vol: Handle<dyn Quote>,
        day_counter: DayCounter,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let surface = ConstantOptionletVolatility::moving_with_quote(
            0,
            NullCalendar::new(),
            BusinessDayConvention::Following,
            vol,
            day_counter,
            VolatilityType::Normal,
            0.0,
            settings,
        );
        let vol = Handle::new(shared(surface) as Shared<dyn OptionletVolatilityStructure>);
        Self::new(discount_curve, vol)
    }
}

impl AsObservable for BachelierCapFloorEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for BachelierCapFloorEngine {
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
        let discount = self.discount_curve.current_link()?;
        let surface = self.vol.current_link()?;
        let today = surface.reference_date()?;
        let settlement = discount.reference_date()?;

        let arguments = self.base.arguments();
        let cap_floor_type = match arguments.cap_floor_type {
            Some(cap_floor_type) => cap_floor_type,
            None => fail!("cap/floor type not set"),
        };
        let has_cap = matches!(cap_floor_type, CapFloorType::Cap | CapFloorType::Collar);
        let has_floor = matches!(cap_floor_type, CapFloorType::Floor | CapFloorType::Collar);

        let n = arguments.end_dates.len();
        let mut values = Vec::with_capacity(n);
        let mut deltas = Vec::with_capacity(n);
        let mut discount_factors = Vec::with_capacity(n);
        let mut atm_forwards: Vec<Real> = arguments
            .forwards
            .iter()
            .map(|f| f.unwrap_or(0.0))
            .collect();
        let mut value = 0.0;
        let mut vega = 0.0;

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let payment_date = arguments.end_dates[i];
            if payment_date <= settlement {
                values.push(0.0);
                deltas.push(0.0);
                discount_factors.push(0.0);
                continue;
            }
            let d = discount.discount_date(payment_date, false)?;
            discount_factors.push(d);
            let accrual_factor =
                arguments.nominals[i] * arguments.gearings[i] * arguments.accrual_times[i];
            let discounted_accrual = d * accrual_factor;
            let Some(forward) = arguments.forwards[i] else {
                values.push(0.0);
                deltas.push(0.0);
                continue;
            };
            atm_forwards[i] = forward;

            let fixing_date = arguments.fixing_dates[i];
            let sqrt_time = if fixing_date > today {
                surface.time_from_reference(fixing_date)?.sqrt()
            } else {
                0.0
            };

            let mut optionlet_value = 0.0;
            let mut optionlet_vega = 0.0;
            let mut optionlet_delta = 0.0;

            if has_cap {
                let strike = arguments.cap_rates[i].expect("cap rate set for cap/collar");
                let mut std_dev = 0.0;
                if sqrt_time > 0.0 {
                    std_dev = surface
                        .black_variance_date(fixing_date, strike, false)?
                        .sqrt();
                    optionlet_vega += bachelier_black_formula_std_dev_derivative(
                        strike,
                        forward,
                        std_dev,
                        discounted_accrual,
                    )? * sqrt_time;
                    optionlet_delta = bachelier_black_formula_asset_itm_probability(
                        OptionType::Call,
                        strike,
                        forward,
                        std_dev,
                    )?;
                }
                optionlet_value += bachelier_black_formula(
                    OptionType::Call,
                    strike,
                    forward,
                    std_dev,
                    discounted_accrual,
                )?;
            }

            if has_floor {
                let strike = arguments.floor_rates[i].expect("floor rate set for floor/collar");
                let mut std_dev = 0.0;
                let mut floorlet_vega = 0.0;
                let mut floorlet_delta = 0.0;
                if sqrt_time > 0.0 {
                    std_dev = surface
                        .black_variance_date(fixing_date, strike, false)?
                        .sqrt();
                    floorlet_vega = bachelier_black_formula_std_dev_derivative(
                        strike,
                        forward,
                        std_dev,
                        discounted_accrual,
                    )? * sqrt_time;
                    floorlet_delta = (OptionType::Put as i32 as Real)
                        * bachelier_black_formula_asset_itm_probability(
                            OptionType::Put,
                            strike,
                            forward,
                            std_dev,
                        )?;
                }
                let floorlet = bachelier_black_formula(
                    OptionType::Put,
                    strike,
                    forward,
                    std_dev,
                    discounted_accrual,
                )?;
                if cap_floor_type == CapFloorType::Floor {
                    optionlet_value = floorlet;
                    optionlet_vega = floorlet_vega;
                    optionlet_delta = floorlet_delta;
                } else {
                    optionlet_value -= floorlet;
                    optionlet_vega -= floorlet_vega;
                    optionlet_delta -= floorlet_delta;
                }
            }

            values.push(optionlet_value);
            deltas.push(optionlet_delta);
            value += optionlet_value;
            vega += optionlet_vega;
        }

        drop(discount);
        drop(surface);

        let results = self.base.results_mut();
        results.value = Some(value);
        results.error_estimate = None;
        results.valuation_date = None;
        results
            .additional_results
            .insert("vega".to_string(), shared(vega) as Shared<dyn Any>);
        results.additional_results.insert(
            "optionletsPrice".to_string(),
            shared(values) as Shared<dyn Any>,
        );
        results.additional_results.insert(
            "optionletsDelta".to_string(),
            shared(deltas) as Shared<dyn Any>,
        );
        results.additional_results.insert(
            "optionletsDiscountFactor".to_string(),
            shared(discount_factors) as Shared<dyn Any>,
        );
        results.additional_results.insert(
            "optionletsAtmForward".to_string(),
            shared(atm_forwards) as Shared<dyn Any>,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cashflows::{Coupon, IborLeg};
    use crate::indexes::ibor::Euribor;
    use crate::indexes::interestrateindex::InterestRateIndex;
    use crate::instrument::Instrument;
    use crate::instruments::{CapFloor, SwapType, VanillaSwap};
    use crate::interestrate::Compounding;
    use crate::pricingengines::DiscountingSwapEngine;
    use crate::quotes::{Quote, SimpleQuote, make_quote_handle};
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::yields::{FlatForward, ZeroSpreadedTermStructure};
    use crate::time::calendars::target::Target;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::time::schedule::MakeSchedule;
    use crate::time::timeunit::TimeUnit;
    use crate::types::{Rate, Real, Volatility};

    const VOL: Volatility = 0.01;

    fn fixture() -> (
        Shared<Settings<Date>>,
        Handle<dyn YieldTermStructure>,
        Shared<crate::indexes::IborIndex>,
    ) {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(Date::new(14, Month::March, 2002));
        let settlement = Date::new(18, Month::March, 2002);
        let curve: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::with_rate(
            settlement,
            0.05,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        ))
            as Shared<dyn YieldTermStructure>);
        let index = shared(Euribor::six_months(curve.clone(), Shared::clone(&settings)));
        (settings, curve, index)
    }

    fn engine(
        curve: &Handle<dyn YieldTermStructure>,
        settings: &Shared<Settings<Date>>,
        vol: Volatility,
    ) -> SharedMut<dyn PricingEngine> {
        shared_mut(
            BachelierCapFloorEngine::with_flat_vol(
                curve.clone(),
                make_quote_handle(vol).handle(),
                Actual365Fixed::new(),
                Shared::clone(settings),
            )
            .unwrap(),
        ) as SharedMut<dyn PricingEngine>
    }

    fn leg5(
        index: &Shared<crate::indexes::IborIndex>,
        start: Date,
    ) -> Vec<Shared<crate::cashflows::IborCoupon>> {
        let end = Target::new().advance(
            start,
            5,
            TimeUnit::Years,
            BusinessDayConvention::ModifiedFollowing,
            false,
        );
        let schedule = MakeSchedule::new()
            .from(start)
            .to(end)
            .with_frequency(Frequency::Semiannual)
            .with_calendar(Target::new())
            .with_convention(BusinessDayConvention::ModifiedFollowing)
            .with_termination_date_convention(BusinessDayConvention::ModifiedFollowing)
            .forwards()
            .end_of_month(false)
            .build();
        IborLeg::new(schedule, Shared::clone(index))
            .with_notional(100.0)
            .with_payment_day_counter(index.day_counter().clone())
            .with_payment_adjustment(BusinessDayConvention::ModifiedFollowing)
            .with_fixing_days(2)
            .coupons()
            .unwrap()
    }

    fn priced_cap_floor(
        settings: &Shared<Settings<Date>>,
        curve: &Handle<dyn YieldTermStructure>,
        coupons: &[Shared<crate::cashflows::IborCoupon>],
        is_cap: bool,
        strike: Rate,
        vol: Volatility,
    ) -> CapFloor {
        let mut cf = if is_cap {
            CapFloor::cap(coupons.to_vec(), vec![strike], Shared::clone(settings))
        } else {
            CapFloor::floor(coupons.to_vec(), vec![strike], Shared::clone(settings))
        }
        .unwrap();
        cf.base_mut()
            .set_pricing_engine(engine(curve, settings, vol));
        cf
    }

    #[test]
    fn optionlets_price_sums_to_npv_and_vega_matches_fd() {
        let (settings, curve, index) = fixture();
        let start = curve.current_link().unwrap().reference_date().unwrap();
        let coupons = leg5(&index, start);
        let mut cap = priced_cap_floor(&settings, &curve, &coupons, true, 0.05, VOL);
        let prices = cap.result::<Vec<Real>>("optionletsPrice").unwrap();
        assert_eq!(prices.len(), 10);
        let sum: Real = prices.iter().sum();
        assert!((sum - cap.npv().unwrap()).abs() < 1e-12);

        let shift = 1.0e-8;
        let analytical = cap.result::<Real>("vega").unwrap();
        let up = priced_cap_floor(&settings, &curve, &coupons, true, 0.05, VOL + shift)
            .npv()
            .unwrap();
        let down = priced_cap_floor(&settings, &curve, &coupons, true, 0.05, VOL - shift)
            .npv()
            .unwrap();
        let numerical = (up - down) / (2.0 * shift);
        assert!(((numerical - analytical) / numerical).abs() <= 0.005);
    }

    #[test]
    fn cap_minus_floor_equals_the_underlying_swap() {
        let (settings, curve, index) = fixture();
        let start = curve.current_link().unwrap().reference_date().unwrap();
        let coupons = leg5(&index, start);
        let strike = 0.05;
        let cap = priced_cap_floor(&settings, &curve, &coupons, true, strike, VOL)
            .npv()
            .unwrap();
        let floor = priced_cap_floor(&settings, &curve, &coupons, false, strike, VOL)
            .npv()
            .unwrap();
        let end = Target::new().advance(
            start,
            5,
            TimeUnit::Years,
            BusinessDayConvention::ModifiedFollowing,
            false,
        );
        let schedule = MakeSchedule::new()
            .from(start)
            .to(end)
            .with_frequency(Frequency::Semiannual)
            .with_calendar(Target::new())
            .with_convention(BusinessDayConvention::ModifiedFollowing)
            .with_termination_date_convention(BusinessDayConvention::ModifiedFollowing)
            .forwards()
            .end_of_month(false)
            .build();
        let mut swap = VanillaSwap::new(
            SwapType::Payer,
            100.0,
            schedule.clone(),
            strike,
            index.day_counter().clone(),
            schedule,
            Shared::clone(&index),
            0.0,
            index.day_counter().clone(),
            None,
            Shared::clone(&settings),
        )
        .unwrap();
        swap.base_mut()
            .set_pricing_engine(shared_mut(DiscountingSwapEngine::new(
                curve, None, None, None, settings,
            )) as SharedMut<dyn PricingEngine>);
        assert!(((cap - floor) - swap.npv().unwrap()).abs() <= 1.0e-10);
    }

    /// `testBachelierOptionLetsDelta` (`capfloor.cpp:766`): analytic Normal
    /// `optionletsDelta` vs forward-rate FD @ 1e-6 (vol 1%).
    #[test]
    fn optionlets_delta_matches_forward_fd() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(Date::new(14, Month::March, 2002));
        let settlement = Date::new(18, Month::March, 2002);
        let base: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::with_rate(
            settlement,
            0.05,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        ))
            as Shared<dyn YieldTermStructure>);
        let spread = shared(SimpleQuote::new(0.0));
        let spread_h: Handle<dyn Quote> = Handle::new(Shared::clone(&spread) as Shared<dyn Quote>);
        let curve: Handle<dyn YieldTermStructure> =
            Handle::new(shared(ZeroSpreadedTermStructure::new(base, spread_h))
                as Shared<dyn YieldTermStructure>);

        let index = shared(Euribor::six_months(curve.clone(), Shared::clone(&settings)));
        let calendar = Target::new();
        let start = curve.current_link().unwrap().reference_date().unwrap();
        let end = calendar.advance(
            start,
            20,
            TimeUnit::Years,
            BusinessDayConvention::ModifiedFollowing,
            false,
        );
        let schedule = MakeSchedule::new()
            .from(start)
            .to(end)
            .with_frequency(Frequency::Semiannual)
            .with_calendar(calendar)
            .with_convention(BusinessDayConvention::ModifiedFollowing)
            .with_termination_date_convention(BusinessDayConvention::ModifiedFollowing)
            .forwards()
            .end_of_month(false)
            .build();
        let leg = IborLeg::new(schedule, Shared::clone(&index))
            .with_notional(100.0)
            .with_payment_day_counter(index.day_counter().clone())
            .with_payment_adjustment(BusinessDayConvention::ModifiedFollowing)
            .with_fixing_days(2)
            .coupons()
            .unwrap();

        let engine = || {
            shared_mut(
                BachelierCapFloorEngine::with_flat_vol(
                    curve.clone(),
                    make_quote_handle(0.01).handle(),
                    Actual365Fixed::new(),
                    Shared::clone(&settings),
                )
                .unwrap(),
            ) as SharedMut<dyn PricingEngine>
        };
        let mut cap = CapFloor::cap(leg.clone(), vec![0.05], Shared::clone(&settings)).unwrap();
        cap.base_mut().set_pricing_engine(engine());
        let mut floor = CapFloor::floor(leg.clone(), vec![0.05], Shared::clone(&settings)).unwrap();
        floor.base_mut().set_pricing_engine(engine());

        let cap_analytic = cap.result::<Vec<Real>>("optionletsDelta").unwrap();
        let floor_analytic = floor.result::<Vec<Real>>("optionletsDelta").unwrap();

        let eps = 1.0e-6;
        spread.set_value(Some(eps));
        let cap_up_p = cap.result::<Vec<Real>>("optionletsPrice").unwrap();
        let cap_up_d = cap.result::<Vec<Real>>("optionletsDiscountFactor").unwrap();
        let cap_up_f = cap.result::<Vec<Real>>("optionletsAtmForward").unwrap();
        let floor_up_p = floor.result::<Vec<Real>>("optionletsPrice").unwrap();
        let floor_up_d = floor
            .result::<Vec<Real>>("optionletsDiscountFactor")
            .unwrap();
        let floor_up_f = floor.result::<Vec<Real>>("optionletsAtmForward").unwrap();

        spread.set_value(Some(-eps));
        let cap_dn_p = cap.result::<Vec<Real>>("optionletsPrice").unwrap();
        let cap_dn_d = cap.result::<Vec<Real>>("optionletsDiscountFactor").unwrap();
        let cap_dn_f = cap.result::<Vec<Real>>("optionletsAtmForward").unwrap();
        let floor_dn_p = floor.result::<Vec<Real>>("optionletsPrice").unwrap();
        let floor_dn_d = floor
            .result::<Vec<Real>>("optionletsDiscountFactor")
            .unwrap();
        let floor_dn_f = floor.result::<Vec<Real>>("optionletsAtmForward").unwrap();

        assert_eq!(
            cap_analytic[0], 0.0,
            "in-progress first caplet must keep analytic δ=0 (sqrtTime==0)"
        );

        let mut cap_n = 0usize;
        for n in 1..cap_up_p.len() {
            let accrual = leg[n].nominal() * leg[n].accrual_period() * leg[n].gearing();
            let df_fwd = cap_up_f[n] - cap_dn_f[n];
            assert!(
                cap_up_d[n] > 0.0 && cap_dn_d[n] > 0.0 && df_fwd.abs() > 1e-14,
                "caplet {n}: spread bump did not move discount/forward (vacuous FD)"
            );
            let fd = (cap_up_p[n] / cap_up_d[n] - cap_dn_p[n] / cap_dn_d[n]) / df_fwd / accrual;
            assert!(
                (cap_analytic[n] - fd).abs() <= 1.0e-6,
                "caplet {n}: analytic {} vs fd {fd}",
                cap_analytic[n]
            );
            cap_n += 1;
        }
        let mut floor_n = 0usize;
        for n in 0..floor_up_p.len() {
            let accrual = leg[n].nominal() * leg[n].accrual_period() * leg[n].gearing();
            let df_fwd = floor_up_f[n] - floor_dn_f[n];
            if floor_up_d[n] == 0.0 && floor_dn_d[n] == 0.0 {
                assert_eq!(floor_analytic[n], 0.0);
                continue;
            }
            assert!(
                floor_up_d[n] > 0.0 && floor_dn_d[n] > 0.0 && df_fwd.abs() > 1e-14,
                "floorlet {n}: spread bump did not move discount/forward (vacuous FD)"
            );
            let fd =
                (floor_up_p[n] / floor_up_d[n] - floor_dn_p[n] / floor_dn_d[n]) / df_fwd / accrual;
            assert!(
                (floor_analytic[n] - fd).abs() <= 1.0e-6,
                "floorlet {n}: analytic {} vs fd {fd}",
                floor_analytic[n]
            );
            floor_n += 1;
        }
        assert!(
            cap_n >= 39 && floor_n >= 40,
            "too few live optionlets compared: cap {cap_n} floor {floor_n}"
        );
    }
}
