//! Cap/floor on a short-rate lattice.
//!
//! Port of `ql/pricingengines/capfloor/discretizedcapfloor.{hpp,cpp}`:
//! at each optionlet start, prices a discount-bond put/call (cap/floor).
//!
//! The past-start intrinsic path (`post_adjust_values_impl`) matches QL but is
//! unreachable through [`TreeCapFloorEngine`](super::TreeCapFloorEngine)'s
//! time-steps ctor: `mandatory_times` still includes negative starts (QL
//! parity), and `TimeGrid::with_mandatory_times` rejects them. A fixed-grid
//! ctor (deferred) is the escape hatch.

use crate::discretizedasset::{DiscretizedAsset, DiscretizedAssetBase, DiscretizedDiscountBond};
use crate::errors::QlResult;
use crate::fail;
use crate::instruments::{CapFloorArguments, CapFloorType};
use crate::math::array::Array;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::types::{Real, Size, Time};

/// Cap/floor discretized on a lattice (`discretizedcapfloor.hpp`).
pub struct DiscretizedCapFloor {
    base: DiscretizedAssetBase,
    arguments: CapFloorArguments,
    start_times: Vec<Time>,
    end_times: Vec<Time>,
}

impl DiscretizedCapFloor {
    /// `DiscretizedCapFloor(args, referenceDate, dayCounter)`.
    pub fn new(args: CapFloorArguments, reference_date: Date, day_counter: &DayCounter) -> Self {
        let start_times = args
            .start_dates
            .iter()
            .map(|&d| day_counter.year_fraction(reference_date, d))
            .collect();
        let end_times = args
            .end_dates
            .iter()
            .map(|&d| day_counter.year_fraction(reference_date, d))
            .collect();
        Self {
            base: DiscretizedAssetBase::default(),
            arguments: args,
            start_times,
            end_times,
        }
    }
}

impl DiscretizedAsset for DiscretizedCapFloor {
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
        let mut times = self.start_times.clone();
        times.extend_from_slice(&self.end_times);
        times
    }

    fn pre_adjust_values_impl(&mut self) -> QlResult<()> {
        let method = self.require_method()?;
        let time = self.time();
        let n = self.start_times.len();
        let Some(cap_floor_type) = self.arguments.cap_floor_type else {
            fail!("cap/floor type not set");
        };

        for i in 0..n {
            if !self.is_on_time(self.start_times[i]) {
                continue;
            }
            let end = self.end_times[i];
            let tenor = self.arguments.accrual_times[i];
            let gearing = self.arguments.gearings[i];
            let nominal = self.arguments.nominals[i];

            let mut bond = DiscretizedDiscountBond::new();
            bond.initialize(Shared::clone(&method), end)?;
            bond.rollback(time)?;

            let has_cap = matches!(cap_floor_type, CapFloorType::Cap | CapFloorType::Collar);
            let has_floor = matches!(cap_floor_type, CapFloorType::Floor | CapFloorType::Collar);

            if has_cap {
                let Some(cap) = self.arguments.cap_rates[i] else {
                    fail!("cap rate not set for a cap/collar");
                };
                let accrual = 1.0 + cap * tenor;
                let strike = 1.0 / accrual;
                let values = self.values_mut();
                for j in 0..values.size() {
                    values[j] += nominal * accrual * gearing * (strike - bond.values()[j]).max(0.0);
                }
            }
            if has_floor {
                let Some(floor) = self.arguments.floor_rates[i] else {
                    fail!("floor rate not set for a floor/collar");
                };
                let accrual = 1.0 + floor * tenor;
                let strike = 1.0 / accrual;
                let mult: Real = if cap_floor_type == CapFloorType::Floor {
                    1.0
                } else {
                    -1.0
                };
                let values = self.values_mut();
                for j in 0..values.size() {
                    values[j] +=
                        nominal * accrual * mult * gearing * (bond.values()[j] - strike).max(0.0);
                }
            }
        }
        Ok(())
    }

    fn post_adjust_values_impl(&mut self) -> QlResult<()> {
        let n = self.end_times.len();
        let Some(cap_floor_type) = self.arguments.cap_floor_type else {
            fail!("cap/floor type not set");
        };

        for i in 0..n {
            if !self.is_on_time(self.end_times[i]) || self.start_times[i] >= 0.0 {
                continue;
            }
            let Some(fixing) = self.arguments.forwards[i] else {
                continue;
            };
            let nominal = self.arguments.nominals[i];
            let accrual = self.arguments.accrual_times[i];
            let gearing = self.arguments.gearings[i];
            let scale = accrual * nominal * gearing;

            if matches!(cap_floor_type, CapFloorType::Cap | CapFloorType::Collar) {
                let Some(cap) = self.arguments.cap_rates[i] else {
                    fail!("cap rate not set for a cap/collar");
                };
                let add = (fixing - cap).max(0.0) * scale;
                let values = self.values_mut();
                for j in 0..values.size() {
                    values[j] += add;
                }
            }
            if matches!(cap_floor_type, CapFloorType::Floor | CapFloorType::Collar) {
                let Some(floor) = self.arguments.floor_rates[i] else {
                    fail!("floor rate not set for a floor/collar");
                };
                let add = (floor - fixing).max(0.0) * scale;
                let values = self.values_mut();
                if cap_floor_type == CapFloorType::Floor {
                    for j in 0..values.size() {
                        values[j] += add;
                    }
                } else {
                    for j in 0..values.size() {
                        values[j] -= add;
                    }
                }
            }
        }
        Ok(())
    }
}
