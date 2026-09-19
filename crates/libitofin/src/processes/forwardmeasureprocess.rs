//! Forward-measure stochastic processes.
//!
//! Port of `ql/processes/forwardmeasureprocess.{hpp,cpp}`: a 1-D process whose
//! dynamics are under a `T`-forward measure.

use crate::types::Time;

/// 1-D process in a `T`-forward measure (`forwardmeasureprocess.hpp:55`).
pub trait ForwardMeasureProcess1D {
    /// Forward-measure horizon `T`.
    fn forward_measure_time(&self) -> Time;

    /// Sets the forward-measure horizon (`setForwardMeasureTime`).
    fn set_forward_measure_time(&mut self, t: Time);
}

/// Owned measure-time state concrete forward processes embed.
#[derive(Clone, Debug)]
pub struct ForwardMeasureTime {
    t: Time,
}

impl ForwardMeasureTime {
    /// Measure time `T`.
    pub fn new(t: Time) -> Self {
        Self { t }
    }

    /// Current measure time.
    pub fn get(&self) -> Time {
        self.t
    }

    /// Replace the measure time.
    pub fn set(&mut self, t: Time) {
        self.t = t;
    }
}
