//! Piecewise Gaussian short-rate integrals.
//!
//! Port of `ql/processes/gsrprocesscore.cpp`. QuantLib caches the integrals;
//! this port recomputes them. Reversions with absolute value below `1e-4` use
//! the zero-reversion formulas, matching QuantLib's `flushCache` cutoff.

use crate::types::{Real, Time};

const REV_ZERO: Real = 1.0e-4;

pub struct GsrProcessCore {
    times: Vec<Time>,
    vols: Vec<Real>,
    reversions: Vec<Real>,
    rev_zero: Vec<bool>,
}

impl GsrProcessCore {
    pub fn new(times: Vec<Time>, vols: Vec<Real>, reversions: Vec<Real>) -> Self {
        let rev_zero = reversions.iter().map(|a| a.abs() < REV_ZERO).collect();
        Self {
            times,
            vols,
            reversions,
            rev_zero,
        }
    }

    pub fn sigma(&self, t: Time) -> Real {
        self.vol(self.lower_index(t))
    }

    pub fn reversion(&self, t: Time) -> Real {
        self.rev(self.lower_index(t))
    }

    pub fn expectation(&self, w: Time, xw: Real, dt: Time, t_measure: Time) -> Real {
        self.expectation_x0(w, xw, dt, t_measure)
            + self.expectation_rn(w, dt, t_measure)
            + self.expectation_tf(w, dt, t_measure)
    }

    pub fn variance(&self, w: Time, dt: Time, t_measure: Time) -> Real {
        let t = w + dt;
        let mut res = 0.0;
        for k in incl(self.lower_index(w), self.upper_index(t) - 1) {
            let mut res2 = self.vol(k) * self.vol(k);
            res2 *= if self.rev_zero(k) {
                -(self.floored_time(k, w, t_measure) - self.capped_time(k + 1, t, t_measure))
            } else {
                (1.0 - (2.0
                    * self.rev(k)
                    * (self.floored_time(k, w, t_measure) - self.capped_time(k + 1, t, t_measure)))
                .exp())
                    / (2.0 * self.rev(k))
            };
            for i in incl(k + 1, self.upper_index(t) - 1) {
                res2 *= (-2.0
                    * self.rev(i)
                    * (self.capped_time(i + 1, t, t_measure) - self.time2(i, t_measure)))
                .exp();
            }
            res += res2;
        }
        res
    }

    pub fn y(&self, t: Time, t_measure: Time) -> Real {
        let mut res = 0.0;
        for i in incl(0, self.upper_index(t) - 1) {
            let mut res2 = 1.0;
            for j in incl(i + 1, self.upper_index(t) - 1) {
                res2 *= (-2.0
                    * self.rev(j)
                    * (self.capped_time(j + 1, t, t_measure) - self.time2(j, t_measure)))
                .exp();
            }
            res2 *= if self.rev_zero(i) {
                self.vol(i)
                    * self.vol(i)
                    * (self.capped_time(i + 1, t, t_measure) - self.time2(i, t_measure))
            } else {
                self.vol(i) * self.vol(i) / (2.0 * self.rev(i))
                    * (1.0
                        - (-2.0
                            * self.rev(i)
                            * (self.capped_time(i + 1, t, t_measure) - self.time2(i, t_measure)))
                        .exp())
            };
            res += res2;
        }
        res
    }

    pub fn g(&self, t: Time, w: Time, t_measure: Time) -> Real {
        let mut res = 0.0;
        for i in incl(self.lower_index(t), self.upper_index(w) - 1) {
            let mut res2 = 1.0;
            for j in incl(self.lower_index(t), i - 1) {
                res2 *= (-self.rev(j)
                    * (self.time2(j + 1, t_measure) - self.floored_time(j, t, t_measure)))
                .exp();
            }
            res2 *= if self.rev_zero(i) {
                self.capped_time(i + 1, w, t_measure) - self.floored_time(i, t, t_measure)
            } else {
                (1.0 - (-self.rev(i)
                    * (self.capped_time(i + 1, w, t_measure) - self.floored_time(i, t, t_measure)))
                .exp())
                    / self.rev(i)
            };
            res += res2;
        }
        res
    }

    fn expectation_x0(&self, w: Time, xw: Real, dt: Time, t_measure: Time) -> Real {
        let t = w + dt;
        let mut factor = 1.0;
        for i in incl(self.lower_index(w), self.upper_index(t) - 1) {
            factor *= (-self.rev(i)
                * (self.capped_time(i + 1, t, t_measure) - self.floored_time(i, w, t_measure)))
            .exp();
        }
        factor * xw
    }

    fn expectation_rn(&self, w: Time, dt: Time, t_measure: Time) -> Real {
        let t = w + dt;
        let mut res = 0.0;
        for k in incl(self.lower_index(w), self.upper_index(t) - 1) {
            for l in incl(0, k - 1) {
                let mut res2 = if self.rev_zero(l) {
                    self.vol(l)
                        * self.vol(l)
                        * (self.time2(l + 1, t_measure) - self.time2(l, t_measure))
                } else {
                    self.vol(l) * self.vol(l) / (2.0 * self.rev(l))
                        * (1.0
                            - (-2.0
                                * self.rev(l)
                                * (self.time2(l + 1, t_measure) - self.time2(l, t_measure)))
                            .exp())
                };
                for i in incl(k + 1, self.upper_index(t) - 1) {
                    res2 *= (-self.rev(i)
                        * (self.capped_time(i + 1, t, t_measure) - self.time2(i, t_measure)))
                    .exp();
                }
                for j in incl(l + 1, k - 1) {
                    res2 *= (-2.0
                        * self.rev(j)
                        * (self.time2(j + 1, t_measure) - self.time2(j, t_measure)))
                    .exp();
                }
                res2 *= if self.rev_zero(k) {
                    2.0 * self.time2(k, t_measure)
                        - self.floored_time(k, w, t_measure)
                        - self.capped_time(k + 1, t, t_measure)
                        - 2.0 * (self.time2(k, t_measure) - self.capped_time(k + 1, t, t_measure))
                } else {
                    ((self.rev(k)
                        * (2.0 * self.time2(k, t_measure)
                            - self.floored_time(k, w, t_measure)
                            - self.capped_time(k + 1, t, t_measure)))
                    .exp()
                        - (2.0
                            * self.rev(k)
                            * (self.time2(k, t_measure) - self.capped_time(k + 1, t, t_measure)))
                        .exp())
                        / self.rev(k)
                };
                res += res2;
            }
            let mut res2 = if self.rev_zero(k) {
                self.vol(k) * self.vol(k) / 4.0
                    * (4.0
                        * (self.capped_time(k + 1, t, t_measure) - self.time2(k, t_measure))
                            .powi(2)
                        - ((self.floored_time(k, w, t_measure) - 2.0 * self.time2(k, t_measure)
                            + self.capped_time(k + 1, t, t_measure))
                        .powi(2)
                            + (self.capped_time(k + 1, t, t_measure)
                                - self.floored_time(k, w, t_measure))
                            .powi(2)))
            } else {
                self.vol(k) * self.vol(k) / (2.0 * self.rev(k) * self.rev(k))
                    * ((-2.0
                        * self.rev(k)
                        * (self.capped_time(k + 1, t, t_measure) - self.time2(k, t_measure)))
                    .exp()
                        + 1.0
                        - ((-self.rev(k)
                            * (self.floored_time(k, w, t_measure)
                                - 2.0 * self.time2(k, t_measure)
                                + self.capped_time(k + 1, t, t_measure)))
                        .exp()
                            + (-self.rev(k)
                                * (self.capped_time(k + 1, t, t_measure)
                                    - self.floored_time(k, w, t_measure)))
                            .exp()))
            };
            for i in incl(k + 1, self.upper_index(t) - 1) {
                res2 *= (-self.rev(i)
                    * (self.capped_time(i + 1, t, t_measure) - self.time2(i, t_measure)))
                .exp();
            }
            res += res2;
        }
        res
    }

    fn expectation_tf(&self, w: Time, dt: Time, t_measure: Time) -> Real {
        let t = w + dt;
        let mut res = 0.0;
        for k in incl(self.lower_index(w), self.upper_index(t) - 1) {
            let mut res2 = 0.0;
            for l in incl(k + 1, self.upper_index(t_measure) - 1) {
                let mut res3 = if self.rev_zero(l) {
                    self.capped_time(l + 1, t_measure, t_measure) - self.time2(l, t_measure)
                } else {
                    (1.0 - (-self.rev(l)
                        * (self.capped_time(l + 1, t_measure, t_measure)
                            - self.time2(l, t_measure)))
                    .exp())
                        / self.rev(l)
                };
                for i in incl(k + 1, self.upper_index(t) - 1) {
                    res3 *= (-self.rev(i)
                        * (self.capped_time(i + 1, t, t_measure) - self.time2(i, t_measure)))
                    .exp();
                }
                for j in incl(k + 1, l - 1) {
                    res3 *= (-self.rev(j)
                        * (self.time2(j + 1, t_measure) - self.time2(j, t_measure)))
                    .exp();
                }
                res3 *= if self.rev_zero(k) {
                    (self.capped_time(k + 1, t, t_measure)
                        - self.time2(k + 1, t_measure)
                        - (2.0 * self.floored_time(k, w, t_measure)
                            - self.capped_time(k + 1, t, t_measure)
                            - self.time2(k + 1, t_measure)))
                        / 2.0
                } else {
                    ((self.rev(k)
                        * (self.capped_time(k + 1, t, t_measure) - self.time2(k + 1, t_measure)))
                    .exp()
                        - (self.rev(k)
                            * (2.0 * self.floored_time(k, w, t_measure)
                                - self.capped_time(k + 1, t, t_measure)
                                - self.time2(k + 1, t_measure)))
                        .exp())
                        / (2.0 * self.rev(k))
                };
                res2 += res3;
            }
            let mut res3 = if self.rev_zero(k) {
                (-(self.capped_time(k + 1, t, t_measure)
                    - self.capped_time(k + 1, t_measure, t_measure))
                .powi(2)
                    - 2.0
                        * (self.capped_time(k + 1, t, t_measure)
                            - self.floored_time(k, w, t_measure))
                        .powi(2)
                    + (2.0 * self.floored_time(k, w, t_measure)
                        - self.capped_time(k + 1, t_measure, t_measure)
                        - self.capped_time(k + 1, t, t_measure))
                    .powi(2))
                    / 4.0
            } else {
                (2.0 - (self.rev(k)
                    * (self.capped_time(k + 1, t, t_measure)
                        - self.capped_time(k + 1, t_measure, t_measure)))
                .exp()
                    - (2.0
                        * (-self.rev(k)
                            * (self.capped_time(k + 1, t, t_measure)
                                - self.floored_time(k, w, t_measure)))
                        .exp()
                        - (self.rev(k)
                            * (2.0 * self.floored_time(k, w, t_measure)
                                - self.capped_time(k + 1, t_measure, t_measure)
                                - self.capped_time(k + 1, t, t_measure)))
                        .exp()))
                    / (2.0 * self.rev(k) * self.rev(k))
            };
            for i in incl(k + 1, self.upper_index(t) - 1) {
                res3 *= (-self.rev(i)
                    * (self.capped_time(i + 1, t, t_measure) - self.time2(i, t_measure)))
                .exp();
            }
            res2 += res3;
            res += -self.vol(k) * self.vol(k) * res2;
        }
        res
    }

    fn lower_index(&self, t: Time) -> i32 {
        self.times.partition_point(|time| *time <= t) as i32
    }

    fn upper_index(&self, t: Time) -> i32 {
        if t < f64::MIN_POSITIVE {
            return 0;
        }
        self.times.partition_point(|time| *time <= t - f64::EPSILON) as i32 + 1
    }

    fn capped_time(&self, index: i32, cap: Time, t_measure: Time) -> Time {
        cap.min(self.time2(index, t_measure))
    }

    fn floored_time(&self, index: i32, floor: Time, t_measure: Time) -> Time {
        floor.max(self.time2(index, t_measure))
    }

    fn time2(&self, index: i32, t_measure: Time) -> Time {
        if index <= 0 {
            0.0
        } else if index as usize > self.times.len() {
            t_measure
        } else {
            self.times[index as usize - 1]
        }
    }

    fn vol(&self, index: i32) -> Real {
        let index = index.max(0) as usize;
        if index >= self.vols.len() {
            *self.vols.last().unwrap_or(&0.0)
        } else {
            self.vols[index]
        }
    }

    fn rev(&self, index: i32) -> Real {
        let index = index.max(0) as usize;
        if index >= self.reversions.len() {
            *self.reversions.last().unwrap_or(&0.0)
        } else {
            self.reversions[index]
        }
    }

    fn rev_zero(&self, index: i32) -> bool {
        let index = index.max(0) as usize;
        if index >= self.rev_zero.len() {
            self.rev_zero.last().copied().unwrap_or(false)
        } else {
            self.rev_zero[index]
        }
    }
}

fn incl(start: i32, end: i32) -> impl Iterator<Item = i32> {
    if start > end { 0..0 } else { start..(end + 1) }
}
