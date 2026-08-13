//! Discretized convertible bond on a Tsiveriotis–Fernandes lattice.
//!
//! Port of QuantLib's `ql/pricingengines/bond/discretizedconvertible.{hpp,cpp}`.
//! The asset carries conversion probabilities and blended (equity / debt)
//! discount rates; its [`rollback`](DiscretizedAsset::rollback) /
//! [`partial_rollback`](DiscretizedAsset::partial_rollback) override the
//! generic lattice step so those arrays roll back with the bond values.

use crate::cashflows::Dividend;
use crate::discretizedasset::{DiscretizedAsset, DiscretizedAssetBase};
use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::instruments::{CallabilityType, ConvertibleBondArguments};
use crate::math::array::Array;
use crate::math::comparison::close;
use crate::math::timegrid::TimeGrid;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::Shared;
use crate::types::{Real, Size, Time};

/// Dividend schedule attached to the convertible engine.
pub type DividendSchedule = Vec<Shared<dyn Dividend>>;

/// Lattice-rolled convertible bond with TF conversion / credit state.
pub struct DiscretizedConvertible {
    base: DiscretizedAssetBase,
    arguments: ConvertibleBondArguments,
    process: Shared<GeneralizedBlackScholesProcess>,
    credit_spread: Real,
    risk_free_rate: Real,
    pu: Real,
    pd: Real,
    dt: Time,
    conversion_probability: Array,
    spread_adjusted_rate: Array,
    /// Discounted dividend amounts (exposed for oracle tests).
    dividend_values: Array,
    stopping_times: Vec<Time>,
    callability_times: Vec<Time>,
    coupon_times: Vec<Time>,
    coupon_amounts: Vec<Real>,
    dividend_times: Vec<Time>,
    dividends: DividendSchedule,
}

impl DiscretizedConvertible {
    /// Builds the discretized convertible from engine arguments
    /// (`discretizedconvertible.cpp` ctor).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        arguments: ConvertibleBondArguments,
        process: Shared<GeneralizedBlackScholesProcess>,
        dividends: DividendSchedule,
        credit_spread: Real,
        risk_free_rate: Real,
        pu: Real,
        pd: Real,
        dt: Time,
        grid: &TimeGrid,
    ) -> QlResult<Self> {
        let bond_settlement = arguments
            .settlement_date
            .expect("validated settlement date");
        let risk_free = process.risk_free_rate().current_link()?;
        let day_counter = risk_free.require_day_counter()?;

        // Keep dividends that have not occurred as of settlement
        // (`hasOccurred(settlement, false)` → date <= settlement).
        let mut live_dividends = Vec::new();
        let mut dividend_dates = Vec::new();
        for dividend in dividends {
            if dividend.date() > bond_settlement {
                dividend_dates.push(dividend.date());
                live_dividends.push(dividend);
            }
        }

        let mut dividend_values = Array::filled(live_dividends.len(), 0.0);
        for (i, dividend) in live_dividends.iter().enumerate() {
            dividend_values[i] =
                dividend.amount()? * risk_free.discount_date(dividend_dates[i], true)?;
        }

        let exercise = arguments.exercise.as_ref().expect("validated exercise");
        let mut stopping_times = Vec::with_capacity(exercise.dates().len());
        for &date in exercise.dates() {
            stopping_times.push(day_counter.year_fraction(bond_settlement, date));
        }

        let mut callability_times = Vec::with_capacity(arguments.callability_dates.len());
        for &date in &arguments.callability_dates {
            callability_times.push(day_counter.year_fraction(bond_settlement, date));
        }

        let mut coupon_times = Vec::new();
        let mut coupon_amounts = Vec::new();
        if arguments.cashflows.len() > 1 {
            for flow in &arguments.cashflows[..arguments.cashflows.len() - 1] {
                if flow.date() > bond_settlement {
                    coupon_times.push(day_counter.year_fraction(bond_settlement, flow.date()));
                    coupon_amounts.push(flow.amount()?);
                }
            }
        }

        let mut dividend_times = Vec::with_capacity(dividend_dates.len());
        for &date in &dividend_dates {
            dividend_times.push(day_counter.year_fraction(bond_settlement, date));
        }

        if !grid.empty() {
            for t in &mut stopping_times {
                *t = grid.times()[grid.closest_index(*t)];
            }
            for t in &mut coupon_times {
                *t = grid.times()[grid.closest_index(*t)];
            }
            for t in &mut callability_times {
                *t = grid.times()[grid.closest_index(*t)];
            }
            for t in &mut dividend_times {
                *t = grid.times()[grid.closest_index(*t)];
            }
        }

        Ok(Self {
            base: DiscretizedAssetBase::default(),
            arguments,
            process,
            credit_spread,
            risk_free_rate,
            pu,
            pd,
            dt,
            conversion_probability: Array::new(),
            spread_adjusted_rate: Array::new(),
            dividend_values,
            stopping_times,
            callability_times,
            coupon_times,
            coupon_amounts,
            dividend_times,
            dividends: live_dividends,
        })
    }

    /// Conversion-probability grid (for tests / lattice introspection).
    #[allow(dead_code)]
    pub fn conversion_probability(&self) -> &Array {
        &self.conversion_probability
    }

    /// Blended discount-rate grid.
    #[allow(dead_code)]
    pub fn spread_adjusted_rate(&self) -> &Array {
        &self.spread_adjusted_rate
    }

    /// Present values of future dividends as of the process reference.
    pub fn dividend_values(&self) -> &Array {
        &self.dividend_values
    }

    fn adjusted_grid(&self) -> QlResult<Array> {
        let t = self.time();
        let method = self.require_method()?;
        let mut grid = method.grid(t)?;
        let risk_free = self.process.risk_free_rate().current_link()?;
        let discount_t = if close(t, 0.0) {
            1.0
        } else {
            // Times are measured from settlement; map via process.time on
            // settlement + duration is awkward, so discount ratios use the
            // process day-counter year fractions already stored as times
            // against the risk-free curve from its own reference. Match QL:
            // discount(dividendTime) / discount(t) where both are year
            // fractions from bond settlement expressed with the RF day count.
            // QL uses process_->riskFreeRate()->discount(Time) with those
            // year fractions as if they were curve times — which is correct
            // only when the curve reference equals settlement. The engine
            // flattens curves at the process reference; tests align settlement
            // with that reference (settlementDays=0) or accept the same
            // approximation QuantLib makes.
            risk_free.discount(t, true)?
        };
        for (i, dividend) in self.dividends.iter().enumerate() {
            let dividend_time = self.dividend_times[i];
            if dividend_time >= t || close(dividend_time, t) {
                let dividend_discount = risk_free.discount(dividend_time, true)? / discount_t;
                for j in 0..grid.size() {
                    grid[j] += dividend.amount_with_underlying(grid[j]) * dividend_discount;
                }
            }
        }
        Ok(grid)
    }

    fn apply_convertibility(&mut self) -> QlResult<()> {
        let grid = self.adjusted_grid()?;
        let ratio = self.arguments.conversion_ratio;
        for j in 0..self.values().size() {
            let payoff = ratio * grid[j];
            if self.values()[j] <= payoff {
                self.values_mut()[j] = payoff;
                self.conversion_probability[j] = 1.0;
            }
        }
        Ok(())
    }

    fn apply_callability(&mut self, i: Size, convertible: bool) -> QlResult<()> {
        let grid = self.adjusted_grid()?;
        let price = self.arguments.callability_prices[i];
        let ratio = self.arguments.conversion_ratio;
        match self.arguments.callability_types[i] {
            CallabilityType::Call => {
                if let Some(trigger_mult) = self.arguments.callability_triggers[i] {
                    let conversion_value = self.arguments.redemption / ratio;
                    let trigger = conversion_value * trigger_mult;
                    for j in 0..self.values().size() {
                        if grid[j] >= trigger {
                            self.values_mut()[j] = self.values()[j].min(price.max(ratio * grid[j]));
                        }
                    }
                } else if convertible {
                    for j in 0..self.values().size() {
                        self.values_mut()[j] = self.values()[j].min(price.max(ratio * grid[j]));
                    }
                } else {
                    for j in 0..self.values().size() {
                        self.values_mut()[j] = self.values()[j].min(price);
                    }
                }
            }
            CallabilityType::Put => {
                for j in 0..self.values().size() {
                    self.values_mut()[j] = self.values()[j].max(price);
                }
            }
        }
        Ok(())
    }

    fn add_coupon(&mut self, i: Size) {
        let amount = self.coupon_amounts[i];
        let values = self.values_mut();
        for j in 0..values.size() {
            values[j] += amount;
        }
    }

    /// One TF backward step (`tflattice.hpp` `stepback`).
    fn tf_stepback(
        &self,
        values: &Array,
        conversion_probability: &Array,
        spread_adjusted_rate: &Array,
        new_values: &mut Array,
        new_conversion_probability: &mut Array,
        new_spread_adjusted_rate: &mut Array,
    ) {
        let size = new_values.size();
        for j in 0..size {
            new_conversion_probability[j] =
                self.pd * conversion_probability[j] + self.pu * conversion_probability[j + 1];
            new_spread_adjusted_rate[j] = new_conversion_probability[j] * self.risk_free_rate
                + (1.0 - new_conversion_probability[j])
                    * (self.risk_free_rate + self.credit_spread);
            new_values[j] = self.pd * values[j] / (1.0 + spread_adjusted_rate[j] * self.dt)
                + self.pu * values[j + 1] / (1.0 + spread_adjusted_rate[j + 1] * self.dt);
        }
    }
}

impl DiscretizedAsset for DiscretizedConvertible {
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
        *self.values_mut() = Array::filled(size, self.arguments.redemption);
        self.conversion_probability = Array::filled(size, 0.0);
        self.spread_adjusted_rate = Array::filled(size, 0.0);
        self.adjust_values()?;

        for j in 0..size {
            self.spread_adjusted_rate[j] = self.conversion_probability[j] * self.risk_free_rate
                + (1.0 - self.conversion_probability[j])
                    * (self.risk_free_rate + self.credit_spread);
        }
        Ok(())
    }

    fn mandatory_times(&self) -> Vec<Time> {
        let mut result = Vec::new();
        result.extend_from_slice(&self.stopping_times);
        result.extend_from_slice(&self.callability_times);
        result.extend_from_slice(&self.coupon_times);
        result
    }

    fn post_adjust_values_impl(&mut self) -> QlResult<()> {
        let exercise = self
            .arguments
            .exercise
            .as_ref()
            .expect("validated exercise");
        let mut convertible = false;
        match exercise.exercise_type() {
            ExerciseType::American => {
                if self.time() <= self.stopping_times[1] && self.time() >= self.stopping_times[0] {
                    convertible = true;
                }
            }
            ExerciseType::European => {
                if self.is_on_time(self.stopping_times[0]) {
                    convertible = true;
                }
            }
            ExerciseType::Bermudan => {
                for &stopping_time in &self.stopping_times {
                    if self.is_on_time(stopping_time) {
                        convertible = true;
                    }
                }
            }
        }

        for i in 0..self.callability_times.len() {
            if self.is_on_time(self.callability_times[i]) {
                self.apply_callability(i, convertible)?;
            }
        }
        for i in 0..self.coupon_times.len() {
            if self.is_on_time(self.coupon_times[i]) {
                self.add_coupon(i);
            }
        }
        if convertible {
            self.apply_convertibility()?;
        }
        Ok(())
    }

    fn rollback(&mut self, to: Time) -> QlResult<()> {
        self.partial_rollback(to)?;
        self.adjust_values()
    }

    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn partial_rollback(&mut self, to: Time) -> QlResult<()> {
        let from = self.time();
        if close(from, to) {
            return Ok(());
        }
        require!(
            from > to,
            "cannot roll the asset back to {to} (it is already at t = {from})"
        );
        let method = self.require_method()?;
        let grid = method.time_grid().clone();
        let i_from = grid.index(from)?;
        let i_to = grid.index(to)?;

        for i in (i_to..i_from).rev() {
            let values = std::mem::replace(self.values_mut(), Array::new());
            let conversion_probability =
                std::mem::replace(&mut self.conversion_probability, Array::new());
            let spread_adjusted_rate =
                std::mem::replace(&mut self.spread_adjusted_rate, Array::new());
            let mut new_values = Array::filled(i + 1, 0.0);
            let mut new_conversion_probability = Array::filled(i + 1, 0.0);
            let mut new_spread_adjusted_rate = Array::filled(i + 1, 0.0);
            self.tf_stepback(
                &values,
                &conversion_probability,
                &spread_adjusted_rate,
                &mut new_values,
                &mut new_conversion_probability,
                &mut new_spread_adjusted_rate,
            );
            self.set_time(grid[i]);
            *self.values_mut() = new_values;
            self.conversion_probability = new_conversion_probability;
            self.spread_adjusted_rate = new_spread_adjusted_rate;
            if i != i_to {
                self.adjust_values()?;
            }
        }
        Ok(())
    }

    fn present_value(&mut self) -> QlResult<Real> {
        // At the root there is a single node; Arrow-Debreu state price is 1.
        require!(self.values().size() > 0, "no values to present-value");
        Ok(self.values()[0])
    }
}
