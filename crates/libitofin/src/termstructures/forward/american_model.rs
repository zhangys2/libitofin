//! Settings and reporting types for the model-based American forward path.
//!
//! The model path de-Americanizes option quotes under a trial carry before running the
//! parity estimator, so it needs a few inputs the European paths do not: an optional
//! discrete cash-dividend schedule, the carry search and plausibility ranges, and a work
//! budget. The result reports which tier produced the forward, orthogonal flags, and
//! numeric diagnostics.

use crate::errors::QlResult;
use crate::require;
use crate::types::{Rate, Real, Size, Time};

/// Accuracy level of the model-based path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AmericanLevel {
    /// `Accurate` when a dividend schedule is supplied, otherwise `Fast`.
    #[default]
    Auto,
    /// Closed-form pricer only; a supplied dividend schedule is ignored and flagged.
    Fast,
    /// Closed-form solve followed by premium-correction passes with a lattice or grid pricer.
    Accurate,
}

/// Which estimator produced the forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardTier {
    /// Model-free strike corridor around the money.
    Corridor,
    /// Model-based, closed-form American pricer.
    ModelFast,
    /// Model-based, closed-form solve plus premium correction.
    ModelAccurate,
}

impl ForwardTier {
    /// Stable lowercase name used by the bindings.
    pub fn as_str(self) -> &'static str {
        match self {
            ForwardTier::Corridor => "corridor",
            ForwardTier::ModelFast => "model_fast",
            ForwardTier::ModelAccurate => "model_accurate",
        }
    }
}

/// Independent boolean signals about how an American forward was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ForwardFlags(u8);

impl ForwardFlags {
    /// No flag set.
    pub const NONE: ForwardFlags = ForwardFlags(0);
    /// A dividend schedule was supplied but the Fast level ignored it.
    pub const SCHEDULE_IGNORED: ForwardFlags = ForwardFlags(1);
    /// The work budget was exceeded and the correction was reduced or dropped.
    pub const LATENCY_FALLBACK: ForwardFlags = ForwardFlags(1 << 1);
    /// The carry or correction iteration did not reach its tolerance.
    pub const NON_CONVERGED: ForwardFlags = ForwardFlags(1 << 2);

    const NAMED: [(ForwardFlags, &'static str); 3] = [
        (ForwardFlags::SCHEDULE_IGNORED, "schedule_ignored"),
        (ForwardFlags::LATENCY_FALLBACK, "latency_fallback"),
        (ForwardFlags::NON_CONVERGED, "non_converged"),
    ];

    /// Raw bits, for the C ABI.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// True when no flag is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// True when every flag in `other` is set.
    pub const fn contains(self, other: ForwardFlags) -> bool {
        self.0 & other.0 == other.0
    }

    /// Sets every flag in `other`.
    pub fn insert(&mut self, other: ForwardFlags) {
        self.0 |= other.0;
    }

    /// Stable lowercase names of the set flags, in declaration order.
    pub fn names(self) -> Vec<&'static str> {
        Self::NAMED
            .iter()
            .filter(|(flag, _)| self.contains(*flag))
            .map(|(_, name)| *name)
            .collect()
    }
}

impl std::ops::BitOr for ForwardFlags {
    type Output = ForwardFlags;

    fn bitor(self, rhs: ForwardFlags) -> ForwardFlags {
        ForwardFlags(self.0 | rhs.0)
    }
}

/// Inputs of the model-based American path. Ignored for the other conventions.
#[derive(Debug, Clone, PartialEq)]
pub struct AmericanModelConfig {
    /// Accuracy level; see [`AmericanLevel`].
    pub level: AmericanLevel,
    /// Discrete cash dividends as `(time in years, amount)`, on the same time basis as
    /// `expiry_years`. Each time must lie in `(0, expiry]` and each amount must be non-negative.
    pub dividends: Vec<(Time, Real)>,
    /// Strike window of the model path, in standard deviations: strikes with
    /// `|ln(K/S)| <= corridor_kappa * sigma_atm * sqrt(T)` are used (at least 0.05 in log
    /// moneyness). Wider than the corridor tier's `american_kappa`, since the model removes
    /// the early-exercise premium instead of avoiding it.
    pub corridor_kappa: Real,
    /// Interval searched for the solved carry (continuous yield), per year.
    pub carry_search_range: (Rate, Rate),
    /// Solved carries outside this interval are reported with `WarningImplausibleCarry`.
    /// Must lie inside `carry_search_range`.
    pub carry_flag_range: (Rate, Rate),
    /// Premium-correction passes; `None` lets the level decide (one, or two with a schedule).
    pub correction_passes: Option<Size>,
    /// Absolute tolerance on the solved carry.
    pub carry_tolerance: Real,
    /// Work budget: the most accurate-pricer evaluations a solve may spend on correction.
    pub max_accurate_prices: Size,
}

impl Default for AmericanModelConfig {
    fn default() -> Self {
        Self {
            level: AmericanLevel::Auto,
            dividends: Vec::new(),
            corridor_kappa: 1.0,
            carry_search_range: (-0.10, 0.25),
            carry_flag_range: (-0.05, 0.15),
            correction_passes: None,
            carry_tolerance: 1e-7,
            max_accurate_prices: 100,
        }
    }
}

/// Largest explicit number of correction passes.
const MAX_CORRECTION_PASSES: Size = 5;

impl AmericanModelConfig {
    /// Checks the settings that do not depend on the expiry.
    pub fn validate(&self) -> QlResult<()> {
        require!(
            self.corridor_kappa.is_finite() && self.corridor_kappa > 0.0,
            "corridor_kappa must be finite and positive, got {}",
            self.corridor_kappa
        );
        let (lo, hi) = self.carry_search_range;
        require!(
            lo.is_finite() && hi.is_finite() && lo < hi,
            "carry_search_range must be a finite increasing interval, got ({lo}, {hi})"
        );
        let (flag_lo, flag_hi) = self.carry_flag_range;
        require!(
            flag_lo.is_finite() && flag_hi.is_finite() && flag_lo < flag_hi,
            "carry_flag_range must be a finite increasing interval, got ({flag_lo}, {flag_hi})"
        );
        require!(
            lo <= flag_lo && flag_hi <= hi,
            "carry_flag_range ({flag_lo}, {flag_hi}) must lie inside carry_search_range ({lo}, {hi})"
        );
        if let Some(passes) = self.correction_passes {
            require!(
                passes <= MAX_CORRECTION_PASSES,
                "correction_passes must be at most {MAX_CORRECTION_PASSES}, got {passes}"
            );
        }
        require!(
            self.carry_tolerance.is_finite() && self.carry_tolerance > 0.0,
            "carry_tolerance must be finite and positive, got {}",
            self.carry_tolerance
        );
        Ok(())
    }

    /// Checks the dividend schedule against the expiry.
    pub fn validate_dividends(&self, expiry_years: Time) -> QlResult<()> {
        for &(time, amount) in &self.dividends {
            require!(
                time.is_finite() && time > 0.0 && time <= expiry_years,
                "dividend time must lie in (0, {expiry_years}], got {time}"
            );
            require!(
                amount.is_finite() && amount >= 0.0,
                "dividend amount must be finite and non-negative, got {amount}"
            );
        }
        Ok(())
    }
}

/// Numeric diagnostics of a model-based solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmericanModelDiagnostics {
    /// Solved continuous yield, or the residual carry on top of the dividend schedule.
    pub solved_carry: Rate,
    /// Mean call implied vol minus put implied vol at the solution.
    pub call_put_iv_gap: Real,
    /// Outer carry iterations spent.
    pub iterations: Size,
    /// Accurate-pricer evaluations spent on correction.
    pub accurate_prices_used: Size,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_agreed_settings() {
        let config = AmericanModelConfig::default();
        assert_eq!(config.corridor_kappa, 1.0);
        assert_eq!(config.level, AmericanLevel::Auto);
        assert!(config.dividends.is_empty());
        assert_eq!(config.carry_search_range, (-0.10, 0.25));
        assert_eq!(config.carry_flag_range, (-0.05, 0.15));
        assert_eq!(config.correction_passes, None);
        assert_eq!(config.max_accurate_prices, 100);
        config.validate().unwrap();
        config.validate_dividends(1.0).unwrap();
    }

    #[test]
    fn validate_rejects_bad_ranges_and_tolerances() {
        let bad = |edit: &dyn Fn(&mut AmericanModelConfig)| {
            let mut config = AmericanModelConfig::default();
            edit(&mut config);
            config.validate().is_err()
        };
        assert!(bad(&|c| c.carry_search_range = (0.25, -0.10)));
        assert!(bad(&|c| c.carry_search_range = (Real::NAN, 0.25)));
        assert!(bad(&|c| c.carry_flag_range = (0.15, -0.05)));
        assert!(bad(&|c| c.carry_flag_range = (-0.20, 0.15)));
        assert!(bad(&|c| c.carry_flag_range = (-0.05, 0.30)));
        assert!(bad(&|c| c.corridor_kappa = 0.0));
        assert!(bad(&|c| c.corridor_kappa = Real::NAN));
        assert!(bad(&|c| c.carry_tolerance = 0.0));
        assert!(bad(&|c| c.carry_tolerance = Real::NAN));
        assert!(bad(&|c| c.correction_passes = Some(6)));
        assert!(!bad(&|c| c.correction_passes = Some(0)));
        assert!(!bad(&|c| c.max_accurate_prices = 0));
    }

    #[test]
    fn dividends_must_lie_inside_the_expiry() {
        let with = |time: Real, amount: Real| AmericanModelConfig {
            dividends: vec![(time, amount)],
            ..Default::default()
        };
        assert!(with(0.25, 1.0).validate_dividends(0.25).is_ok());
        assert!(with(0.25 + 1e-9, 1.0).validate_dividends(0.25).is_err());
        assert!(with(0.0, 1.0).validate_dividends(0.25).is_err());
        assert!(with(-0.1, 1.0).validate_dividends(0.25).is_err());
        assert!(with(0.1, -1.0).validate_dividends(0.25).is_err());
        assert!(with(0.1, Real::NAN).validate_dividends(0.25).is_err());
        assert!(with(Real::NAN, 1.0).validate_dividends(0.25).is_err());
        assert!(with(0.1, 0.0).validate_dividends(0.25).is_ok());
    }

    #[test]
    fn flags_are_independent_and_named_in_order() {
        let mut flags = ForwardFlags::NONE;
        assert!(flags.is_empty());
        assert!(flags.names().is_empty());
        flags.insert(ForwardFlags::NON_CONVERGED);
        flags.insert(ForwardFlags::SCHEDULE_IGNORED);
        assert!(!flags.is_empty());
        assert!(flags.contains(ForwardFlags::SCHEDULE_IGNORED));
        assert!(flags.contains(ForwardFlags::NON_CONVERGED));
        assert!(!flags.contains(ForwardFlags::LATENCY_FALLBACK));
        assert_eq!(flags.names(), vec!["schedule_ignored", "non_converged"]);
        assert_eq!(flags.bits(), 0b101);
        let all = ForwardFlags::SCHEDULE_IGNORED
            | ForwardFlags::LATENCY_FALLBACK
            | ForwardFlags::NON_CONVERGED;
        assert_eq!(
            all.names(),
            vec!["schedule_ignored", "latency_fallback", "non_converged"]
        );
        assert!(all.contains(flags));
    }

    #[test]
    fn tier_names_are_stable() {
        assert_eq!(ForwardTier::Corridor.as_str(), "corridor");
        assert_eq!(ForwardTier::ModelFast.as_str(), "model_fast");
        assert_eq!(ForwardTier::ModelAccurate.as_str(), "model_accurate");
    }
}
