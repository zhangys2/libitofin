//! Brownian-bridge construction of a Wiener path.
//!
//! Port of `ql/methods/montecarlo/brownianbridge.{hpp,cpp}` (Jaeckel, *Monte
//! Carlo Methods in Finance*). The transform maps a vector of unit Gaussians
//! onto a unit-variance increment sequence whose implied path is a Brownian
//! bridge, so the first input drives the terminal value and later inputs fill
//! the interior in dyadic order.
//!
//! Divergences from `brownianbridge.hpp`, all deliberate:
//! - **`transform` is fallible and takes slices**: C++ is a template over
//!   random-access iterators (`brownianbridge.hpp:109`). Slices are the
//!   iterator pair the only consumer ([`PathGenerator`](super::PathGenerator))
//!   holds; a length mismatch is `Err` rather than `QL_REQUIRE`.
//! - **empty input is rejected**: C++ `BrownianBridge(Size steps)` with
//!   `steps == 0` underflows `map[size_-1]`. The path generator never builds
//!   a zero-step grid, so this fails loudly instead of wrapping.

use crate::errors::QlResult;
use crate::math::timegrid::TimeGrid;
use crate::require;
use crate::types::{Real, Size, Time};

/// Builds unit-variance Wiener increments via a Brownian bridge
/// (`brownianbridge.hpp:54`).
pub struct BrownianBridge {
    size: Size,
    t: Vec<Time>,
    sqrtdt: Vec<Time>,
    bridge_index: Vec<Size>,
    left_index: Vec<Size>,
    right_index: Vec<Size>,
    left_weight: Vec<Real>,
    right_weight: Vec<Real>,
    std_dev: Vec<Real>,
}

impl BrownianBridge {
    /// Unit-time steps `1, 2, …, steps` (`brownianbridge.cpp:36`).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `steps` is zero.
    pub fn new(steps: Size) -> QlResult<Self> {
        require!(steps > 0, "at least one step required");
        let t: Vec<Time> = (1..=steps).map(|i| i as Time).collect();
        Self::from_times(t)
    }

    /// Step times copied from `times`. The path is assumed to start at 0,
    /// which must not be included (`brownianbridge.hpp:72`).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `times` is empty.
    pub fn from_times(times: Vec<Time>) -> QlResult<Self> {
        require!(!times.is_empty(), "at least one step required");
        let size = times.len();
        let mut bb = BrownianBridge {
            size,
            t: times,
            sqrtdt: vec![0.0; size],
            bridge_index: vec![0; size],
            left_index: vec![0; size],
            right_index: vec![0; size],
            left_weight: vec![0.0; size],
            right_weight: vec![0.0; size],
            std_dev: vec![0.0; size],
        };
        bb.initialize();
        Ok(bb)
    }

    /// Step times copied from `time_grid[1..]` (`brownianbridge.cpp:52`).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the grid has fewer than two points.
    pub fn from_time_grid(time_grid: &TimeGrid) -> QlResult<Self> {
        require!(time_grid.size() > 1, "at least one step required");
        let t = time_grid.times()[1..].to_vec();
        Self::from_times(t)
    }

    /// Number of increments (`brownianbridge.hpp:81`).
    pub fn size(&self) -> Size {
        self.size
    }

    /// The step times (`brownianbridge.hpp:82`).
    pub fn times(&self) -> &[Time] {
        &self.t
    }

    /// Transforms unit Gaussians `input` into unit-variance increments
    /// (`brownianbridge.hpp:109-140`).
    ///
    /// The first input drives the terminal Brownian value
    /// `W(t_last) = sqrt(t_last) * input[0]`; later inputs fill the bridge.
    /// The output is then differenced and scaled by `1/sqrt(dt)` so a path
    /// generator can feed it to `evolve` as a unit-time Gaussian.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `input` or `output` is not exactly [`size`](Self::size)
    /// long.
    #[allow(clippy::needless_range_loop)]
    pub fn transform(&self, input: &[Real], output: &mut [Real]) -> QlResult<()> {
        require!(input.len() == self.size, "incompatible sequence size");
        require!(output.len() == self.size, "incompatible sequence size");

        output[self.size - 1] = self.std_dev[0] * input[0];
        for i in 1..self.size {
            let j = self.left_index[i];
            let k = self.right_index[i];
            let l = self.bridge_index[i];
            if j != 0 {
                output[l] = self.left_weight[i] * output[j - 1]
                    + self.right_weight[i] * output[k]
                    + self.std_dev[i] * input[i];
            } else {
                output[l] = self.right_weight[i] * output[k] + self.std_dev[i] * input[i];
            }
        }

        for i in (1..self.size).rev() {
            output[i] -= output[i - 1];
            output[i] /= self.sqrtdt[i];
        }
        output[0] /= self.sqrtdt[0];
        Ok(())
    }

    fn initialize(&mut self) {
        self.sqrtdt[0] = self.t[0].sqrt();
        for i in 1..self.size {
            self.sqrtdt[i] = (self.t[i] - self.t[i - 1]).sqrt();
        }

        // map[i] == 0  => path point i is not yet constructed.
        // map[i] > 0   => constructed; map[i]-1 is the constructing variate.
        let mut map = vec![0usize; self.size];
        map[self.size - 1] = 1;
        self.bridge_index[0] = self.size - 1;
        self.std_dev[0] = self.t[self.size - 1].sqrt();
        self.left_weight[0] = 0.0;
        self.right_weight[0] = 0.0;

        let mut j = 0usize;
        for i in 1..self.size {
            while map[j] != 0 {
                j += 1;
            }
            let mut k = j;
            while map[k] == 0 {
                k += 1;
            }
            let l = j + ((k - 1 - j) >> 1);
            map[l] = i;
            self.bridge_index[i] = l;
            self.left_index[i] = j;
            self.right_index[i] = k;
            if j != 0 {
                self.left_weight[i] = (self.t[k] - self.t[l]) / (self.t[k] - self.t[j - 1]);
                self.right_weight[i] = (self.t[l] - self.t[j - 1]) / (self.t[k] - self.t[j - 1]);
                self.std_dev[i] = ((self.t[l] - self.t[j - 1]) * (self.t[k] - self.t[l])
                    / (self.t[k] - self.t[j - 1]))
                    .sqrt();
            } else {
                self.left_weight[i] = (self.t[k] - self.t[l]) / self.t[k];
                self.right_weight[i] = self.t[l] / self.t[k];
                self.std_dev[i] = (self.t[l] * (self.t[k] - self.t[l]) / self.t[k]).sqrt();
            }
            j = k + 1;
            if j >= self.size {
                j = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_step_is_the_identity() {
        // size=1: W(T) = sqrt(T) Z_0, then increment = W(T)/sqrt(T) = Z_0.
        let bb = BrownianBridge::new(1).unwrap();
        let mut out = [0.0];
        bb.transform(&[1.25], &mut out).unwrap();
        assert!((out[0] - 1.25).abs() < 1e-15);
        let grid = TimeGrid::new(2.5, 1).unwrap();
        let bb = BrownianBridge::from_time_grid(&grid).unwrap();
        let mut out = [0.0];
        bb.transform(&[-0.5], &mut out).unwrap();
        assert!((out[0] + 0.5).abs() < 1e-15);
    }

    #[test]
    fn two_step_midpoint_with_zero_interior_variate() {
        // times 0.5, 1.0; input = [1, 0]:
        // W(1) = 1, W(0.5) = 0.5, unit increments both sqrt(1/2).
        let bb = BrownianBridge::from_times(vec![0.5, 1.0]).unwrap();
        assert_eq!(bb.size(), 2);
        assert_eq!(bb.times(), &[0.5, 1.0]);
        let mut out = [0.0, 0.0];
        bb.transform(&[1.0, 0.0], &mut out).unwrap();
        let expected = 0.5_f64.sqrt();
        assert!((out[0] - expected).abs() < 1e-14);
        assert!((out[1] - expected).abs() < 1e-14);
    }

    #[test]
    fn length_mismatch_is_rejected() {
        let bb = BrownianBridge::new(3).unwrap();
        let mut out = [0.0; 3];
        assert!(bb.transform(&[1.0, 2.0], &mut out).is_err());
        let mut short = [0.0; 2];
        assert!(bb.transform(&[1.0, 2.0, 3.0], &mut short).is_err());
    }

    #[test]
    fn empty_construction_is_rejected() {
        assert!(BrownianBridge::new(0).is_err());
        assert!(BrownianBridge::from_times(vec![]).is_err());
        // A default TimeGrid is empty.
        assert!(BrownianBridge::from_time_grid(&TimeGrid::default()).is_err());
    }

    #[test]
    fn from_time_grid_drops_the_origin() {
        let grid = TimeGrid::new(1.0, 4).unwrap();
        let bb = BrownianBridge::from_time_grid(&grid).unwrap();
        assert_eq!(bb.times(), &[0.25, 0.5, 0.75, 1.0]);
    }
}
