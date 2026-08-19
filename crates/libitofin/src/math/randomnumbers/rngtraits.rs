//! Random-number generation policies.
//!
//! Port of `ql/math/randomnumbers/rngtraits.hpp`. QuantLib expresses the
//! sequence-generator and inverse-cumulative concepts through C++ template
//! duck-typing; Rust needs explicit bounds, so this module defines the two
//! concept traits ([`SequenceGenerator`], [`InverseCumulative`]) that the RSG
//! layer is generic over, plus the [`PseudoRandom`] and [`LowDiscrepancy`]
//! policies the Monte Carlo engines consume through [`McRngTraits`].
//!
//! Divergences from `rngtraits.hpp`:
//! - the `static ext::shared_ptr<IC> icInstance` global override hook
//!   (`rngtraits.hpp:58,92`) is dropped (design decision D5, no global
//!   singletons); both policies always default-construct the inverse
//!   cumulative.
//! - the scalar `rng_type` (`InverseCumulativeRng`, `rngtraits.hpp:46`) is
//!   deferred: the path generators consume only the sequence `rsg_type`.
//! - [`SobolRsg`](super::sobol::SobolRsg) does not itself implement
//!   [`SequenceGenerator`] (its inherent `next_sequence` returns a bare
//!   slice); [`LowDiscrepancy`] wraps it in [`SobolSequenceGenerator`] so the
//!   inverse-cumulative adapter sees the same `Sample<Vec<Real>>` surface as
//!   the Mersenne-Twister sequence generator.

use super::inversecumulativersg::InverseCumulativeRsg;
use super::mt19937uniformrng::MersenneTwisterUniformRng;
use super::randomsequencegenerator::RandomSequenceGenerator;
use super::sobol::{DirectionIntegers, PPMT_MAX_DIM, SobolRsg};
use crate::errors::QlResult;
use crate::math::distributions::normal::InverseCumulativeNormal;
use crate::methods::montecarlo::Sample;
use crate::require;
use crate::types::Real;

/// A generator of weighted uniform-or-transformed sequences.
///
/// The Rust bound behind QuantLib's implicit "USG"/"rsg" template concepts
/// (`randomsequencegenerator.hpp:37`, `inversecumulativersg.hpp:42`): a value
/// yielding `dimension` draws as a weighted [`Sample`]. Both
/// [`RandomSequenceGenerator`](super::randomsequencegenerator::RandomSequenceGenerator)
/// and [`InverseCumulativeRsg`](super::inversecumulativersg::InverseCumulativeRsg)
/// implement it.
///
/// `next_sequence` takes `&mut self`, where QuantLib mutates a cached sample
/// through a `const` method; the draw order and values are identical.
pub trait SequenceGenerator {
    /// Advances the generator and returns the freshly drawn sample.
    fn next_sequence(&mut self) -> &Sample<Vec<Real>>;

    /// Returns the most recently drawn sample without advancing.
    fn last_sequence(&self) -> &Sample<Vec<Real>>;

    /// The number of draws per sequence.
    fn dimension(&self) -> usize;
}

/// An inverse cumulative distribution used as a stateless deviate transform.
///
/// The Rust bound behind QuantLib's implicit "IC" template concept
/// (`inversecumulativersg.hpp:50`, `Real IC::operator()(Real)`): map a uniform
/// deviate in `(0, 1)` to the distribution's deviate.
pub trait InverseCumulative {
    /// The distribution deviate for the uniform `x` in `(0, 1)`.
    fn evaluate(&self, x: Real) -> Real;
}

impl InverseCumulative for InverseCumulativeNormal {
    /// # Panics
    ///
    /// The infallible transform boundary: callers of this trait
    /// ([`InverseCumulativeRsg`](super::inversecumulativersg::InverseCumulativeRsg))
    /// feed it uniform deviates that the sequence generator guarantees lie
    /// strictly in `(0, 1)`, where [`InverseCumulativeNormal::value`] is always
    /// finite, so the `expect` never fires. The public [`InverseCumulativeNormal`]
    /// API stays fallible; only this local precondition is asserted here.
    fn evaluate(&self, x: Real) -> Real {
        self.value(x)
            .expect("inverse cumulative normal is finite for a uniform deviate in (0, 1)")
    }
}

/// A Monte Carlo random-number policy: the factory the pricing engines call to
/// build their sequence generator.
///
/// The Rust surface behind QuantLib's `GenericPseudoRandom` traits struct
/// (`rngtraits.hpp:42`). An engine generic over the policy calls
/// [`make_sequence_generator`](McRngTraits::make_sequence_generator) with the
/// path dimensionality and a seed.
pub trait McRngTraits {
    /// The sequence generator this policy builds.
    type RsgType: SequenceGenerator;

    /// Whether the policy supports a Monte Carlo error estimate
    /// (`rngtraits.hpp:50`).
    const ALLOWS_ERROR_ESTIMATE: bool;

    /// Builds a `dimension`-wide sequence generator seeded with `seed`.
    ///
    /// # Errors
    ///
    /// Returns an error if `dimension` is zero.
    fn make_sequence_generator(dimension: usize, seed: u32) -> QlResult<Self::RsgType>;
}

/// Default pseudo-random policy: Mersenne-Twister uniforms mapped through the
/// inverse cumulative normal (`rngtraits.hpp:70`).
pub struct PseudoRandom;

impl McRngTraits for PseudoRandom {
    type RsgType = InverseCumulativeRsg<
        RandomSequenceGenerator<MersenneTwisterUniformRng>,
        InverseCumulativeNormal,
    >;

    const ALLOWS_ERROR_ESTIMATE: bool = true;

    fn make_sequence_generator(dimension: usize, seed: u32) -> QlResult<Self::RsgType> {
        let ursg = RandomSequenceGenerator::with_seed(dimension, seed)?;
        Ok(InverseCumulativeRsg::new(
            ursg,
            InverseCumulativeNormal::standard(),
        ))
    }
}

/// Uniform Sobol sequence as a [`SequenceGenerator`] (`sobolrsg.hpp`).
///
/// [`SobolRsg`](super::sobol::SobolRsg) exposes a bare `&[f64]` draw; this
/// adapter stores the last sample with weight `1.0` so
/// [`InverseCumulativeRsg`] can consume it.
pub struct SobolSequenceGenerator {
    rsg: SobolRsg,
    sample: Sample<Vec<Real>>,
}

impl SobolSequenceGenerator {
    /// Jaeckel-direction Sobol of `dimension`, seeded with `seed`
    /// (`rngtraits.hpp:94`, `SobolRsg(dimension, seed)`).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `dimension` is zero or exceeds [`PPMT_MAX_DIM`].
    pub fn new(dimension: usize, seed: u32) -> QlResult<Self> {
        require!(dimension > 0, "dimensionality must be greater than 0");
        require!(
            dimension <= PPMT_MAX_DIM,
            "dimensionality {dimension} exceeds the number of available primitive polynomials modulo two ({PPMT_MAX_DIM})"
        );
        Ok(SobolSequenceGenerator {
            rsg: SobolRsg::new(dimension, u64::from(seed), DirectionIntegers::Jaeckel),
            sample: Sample::new(vec![0.0; dimension], 1.0),
        })
    }
}

impl SequenceGenerator for SobolSequenceGenerator {
    fn next_sequence(&mut self) -> &Sample<Vec<Real>> {
        let seq = self.rsg.next_sequence();
        self.sample.weight = 1.0;
        self.sample.value.copy_from_slice(seq);
        &self.sample
    }

    fn last_sequence(&self) -> &Sample<Vec<Real>> {
        &self.sample
    }

    fn dimension(&self) -> usize {
        self.rsg.dimension()
    }
}

/// Low-discrepancy policy: Sobol uniforms mapped through the inverse
/// cumulative normal (`rngtraits.hpp:81,103`). Does not support an error
/// estimate (`allowsErrorEstimate = 0`).
pub struct LowDiscrepancy;

impl McRngTraits for LowDiscrepancy {
    type RsgType = InverseCumulativeRsg<SobolSequenceGenerator, InverseCumulativeNormal>;

    const ALLOWS_ERROR_ESTIMATE: bool = false;

    fn make_sequence_generator(dimension: usize, seed: u32) -> QlResult<Self::RsgType> {
        let ursg = SobolSequenceGenerator::new(dimension, seed)?;
        Ok(InverseCumulativeRsg::new(
            ursg,
            InverseCumulativeNormal::standard(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allows_error_estimate<R: McRngTraits>() -> bool {
        R::ALLOWS_ERROR_ESTIMATE
    }

    #[test]
    fn pseudo_random_allows_an_error_estimate() {
        assert!(allows_error_estimate::<PseudoRandom>());
    }

    #[test]
    fn same_seed_generators_are_identical_across_draws() {
        let mut a = PseudoRandom::make_sequence_generator(3, 42).unwrap();
        let mut b = PseudoRandom::make_sequence_generator(3, 42).unwrap();
        for _ in 0..5 {
            assert_eq!(a.next_sequence().value, b.next_sequence().value);
        }
    }

    #[test]
    fn a_different_seed_diverges() {
        let mut a = PseudoRandom::make_sequence_generator(3, 42).unwrap();
        let mut c = PseudoRandom::make_sequence_generator(3, 43).unwrap();
        assert_ne!(a.next_sequence().value, c.next_sequence().value);
    }

    #[test]
    fn low_discrepancy_does_not_allow_an_error_estimate() {
        assert!(!allows_error_estimate::<LowDiscrepancy>());
    }

    #[test]
    fn low_discrepancy_first_dimension_is_van_der_corput_through_the_inverse_normal() {
        // Sobol dim-1 is the van der Corput sequence modulo two; the first
        // draw is 0.5, which the inverse normal maps to 0.
        let mut rsg = LowDiscrepancy::make_sequence_generator(1, 0).unwrap();
        let first = rsg.next_sequence().value[0];
        assert!(
            first.abs() < 1e-15,
            "first inverse-normal Sobol draw {first} should be 0"
        );
    }

    #[test]
    fn low_discrepancy_zero_dimension_is_rejected() {
        assert!(LowDiscrepancy::make_sequence_generator(0, 5).is_err());
    }
}
