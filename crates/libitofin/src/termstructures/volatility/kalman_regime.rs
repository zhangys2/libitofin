//! Volatility-regime detector for scaling the Kalman smile's process noise.
//!
//! The detector watches the ATM volatility and forward of the quoted expiry. A
//! move over a short window that is large relative to the ATM volatility's own
//! recent changes (or, for the forward, relative to what the ATM volatility
//! implies) raises a multiplier `m >= 1`. Pass `dt * m` to
//! [`KalmanSmileSection::predict`](super::KalmanSmileSection::predict): the process
//! noise is linear in elapsed time, so this scales it by `m`. The multiplier jumps
//! up immediately and decays back to 1 exponentially.
//!
//! The detector is opt-in: nothing in the library calls it, and the tuned defaults come
//! from a single market over 15 days.
//!
//! Intended use is a quiet base `process_iv_rate` (about 0.0001) with this
//! detector supplying the extra responsiveness during shocks. Do not combine it
//! with `FilterConfig::adapt_rate`, which scales the same noise term.

use std::collections::VecDeque;

use crate::errors::QlResult;
use crate::require;
use crate::types::{Rate, Real, Time, Volatility};

const SECONDS_PER_YEAR: Real = 365.0 * 86_400.0;

/// The ATM-vol change baseline never falls below this fraction of `initial_change_sd`.
/// Without a floor, a flat or quantized feed decays the baseline to zero and the next
/// ordinary tick looks like an infinite-sigma shock.
const BASELINE_FLOOR_FRACTION: Real = 0.5;

/// Tuning of the [`RegimeDetector`]. All times are in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegimeConfig {
    /// Look-back window over which moves are measured.
    pub window: Time,
    /// z-score above which a move counts as a regime change.
    pub threshold: Real,
    /// Time constant of the exponential decay of the multiplier back to 1.
    pub decay: Time,
    /// Upper bound of the multiplier.
    pub max_multiplier: Real,
    /// Time constant of the slow baseline of window-sized ATM-vol changes.
    pub baseline_tau: Time,
    /// Initial standard deviation of a window-sized ATM-vol change, before the
    /// baseline has learned from data. The learned baseline is floored at half of this.
    pub initial_change_sd: Volatility,
}

impl Default for RegimeConfig {
    /// Values chosen on a 15-day Deribit BTC backtest (window 300 s, z = 3, decay 1200 s).
    ///
    /// With ATM volatility derived from noisy quotes rather than smooth marks, the
    /// detector fires about 1.6x as often at z = 3 for the same fit quality; use
    /// `threshold = 4` there for a comparable duty cycle.
    fn default() -> Self {
        Self {
            window: 300.0,
            threshold: 3.0,
            decay: 1200.0,
            max_multiplier: 100.0,
            baseline_tau: 14_400.0,
            initial_change_sd: 0.001,
        }
    }
}

impl RegimeConfig {
    pub fn validate(&self) -> QlResult<()> {
        for (name, v) in [
            ("window", self.window),
            ("threshold", self.threshold),
            ("decay", self.decay),
            ("baseline_tau", self.baseline_tau),
            ("initial_change_sd", self.initial_change_sd),
        ] {
            require!(
                v.is_finite() && v > 0.0,
                "{name} must be finite and positive"
            );
        }
        require!(
            self.max_multiplier.is_finite() && self.max_multiplier >= 1.0,
            "max_multiplier must be finite and at least 1"
        );
        Ok(())
    }
}

/// Streaming regime detector producing a process-noise multiplier `>= 1`.
#[derive(Clone, Debug)]
pub struct RegimeDetector {
    config: RegimeConfig,
    history: VecDeque<(Time, Volatility, Real)>,
    change_var: Real,
    multiplier: Real,
    last_time: Option<Time>,
}

impl RegimeDetector {
    pub fn new(config: RegimeConfig) -> QlResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            history: VecDeque::new(),
            change_var: config.initial_change_sd * config.initial_change_sd,
            multiplier: 1.0,
            last_time: None,
        })
    }

    /// Current multiplier (1.0 before any observation).
    pub fn multiplier(&self) -> Real {
        self.multiplier
    }

    /// Feeds one observation and returns the updated multiplier.
    ///
    /// `time` is in seconds on any monotone clock; `atm_vol` and `forward` must be
    /// finite and positive. An error leaves the detector unchanged.
    pub fn observe(&mut self, time: Time, atm_vol: Volatility, forward: Rate) -> QlResult<Real> {
        require!(time.is_finite(), "time must be finite");
        require!(
            atm_vol.is_finite() && atm_vol > 0.0,
            "ATM volatility must be finite and positive"
        );
        require!(
            forward.is_finite() && forward > 0.0,
            "forward must be finite and positive"
        );
        if let Some(last) = self.last_time {
            require!(time >= last, "observation times must be non-decreasing");
        }

        let cfg = self.config;
        let dt = self.last_time.map_or(0.0, |last| time - last);
        self.last_time = Some(time);
        let ln_forward = forward.ln();
        self.history.push_back((time, atm_vol, ln_forward));
        // Drop samples older than the window, but never the last reference sample: after a
        // gap longer than the window (or with a sampling interval close to it) the
        // previous sample is the only thing to measure the move against.
        while self.history.len() > 2
            && self
                .history
                .front()
                .is_some_and(|&(t, _, _)| time - t > cfg.window)
        {
            self.history.pop_front();
        }
        let &(t0, vol0, lnf0) = self.history.front().expect("just pushed");
        let span = time - t0;

        let mut z = 0.0;
        let mut dvol = 0.0;
        let measured = span > 0.0;
        // `change_var` is a window-length variance; a random walk's variance is linear in
        // the span it is measured over.
        let span_ratio = span / cfg.window;
        if measured {
            dvol = atm_vol - vol0;
            let forward_sd = atm_vol * (span / SECONDS_PER_YEAR).sqrt();
            let z_forward = ((ln_forward - lnf0) / forward_sd).abs();
            let z_vol = dvol.abs() / (self.change_var * span_ratio).sqrt();
            z = z_forward.max(z_vol);
        }

        let target = if z > cfg.threshold {
            (z / cfg.threshold).powi(2).min(cfg.max_multiplier)
        } else {
            1.0
        };
        self.multiplier = target.max(1.0 + (self.multiplier - 1.0) * (-dt / cfg.decay).exp());

        // Learn the baseline only from unflagged periods so shocks do not raise it.
        if measured && z <= cfg.threshold {
            let a = (dt / cfg.baseline_tau).min(1.0);
            let floor = (BASELINE_FLOOR_FRACTION * cfg.initial_change_sd).powi(2);
            self.change_var =
                ((1.0 - a) * self.change_var + a * dvol * dvol / span_ratio).max(floor);
        }
        Ok(self.multiplier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic small noise in [-1, 1].
    fn wiggle(i: usize) -> Real {
        ((i as Real * 12.9898).sin() * 43758.5453).fract() * 2.0 - 1.0
    }

    fn calm(det: &mut RegimeDetector, start: Time, n: usize, vol: Real, fwd: Real) -> Time {
        let mut t = start;
        for i in 0..n {
            det.observe(
                t,
                vol + 0.0002 * wiggle(i),
                fwd * (1.0 + 1e-5 * wiggle(i + 7)),
            )
            .unwrap();
            t += 10.0;
        }
        t
    }

    #[test]
    fn calm_market_keeps_multiplier_at_one() {
        let mut det = RegimeDetector::new(RegimeConfig::default()).unwrap();
        calm(&mut det, 0.0, 2000, 0.40, 80_000.0);
        assert_eq!(det.multiplier(), 1.0);
    }

    #[test]
    fn vol_shock_raises_then_decays() {
        let mut det = RegimeDetector::new(RegimeConfig::default()).unwrap();
        let mut t = calm(&mut det, 0.0, 1500, 0.40, 80_000.0);
        // ATM vol +3 vp over five minutes, forward flat.
        let mut peak: Real = 1.0;
        for k in 1..=30 {
            peak = peak.max(
                det.observe(t, 0.40 + 0.03 * k as Real / 30.0, 80_000.0)
                    .unwrap(),
            );
            t += 10.0;
        }
        assert!(peak > 10.0, "shock should lift the multiplier, got {peak}");
        assert!(peak <= RegimeConfig::default().max_multiplier);

        // Hold the new level: once the window contains only the new level the move
        // vanishes and the excess multiplier decays as exp(-dt / decay).
        let mut at_40 = 0.0;
        let mut at_150 = 0.0;
        for i in 0..=150 {
            let m = det.observe(t, 0.43, 80_000.0).unwrap();
            if i == 40 {
                at_40 = m;
            }
            if i == 150 {
                at_150 = m;
            }
            t += 10.0;
        }
        assert!(at_40 > 1.5, "{at_40}");
        let ratio = (at_150 - 1.0) / (at_40 - 1.0);
        let expected = (-1100.0_f64 / RegimeConfig::default().decay).exp();
        assert!((ratio - expected).abs() < 1e-9, "{ratio} vs {expected}");
    }

    /// Calm feed sampled every `dt` seconds followed by a single jump of `jump` in ATM vol.
    fn multiplier_after_jump(dt: Time, calm_ticks: usize, gap: Time, jump: Real) -> Real {
        let mut det = RegimeDetector::new(RegimeConfig::default()).unwrap();
        let mut t = 0.0;
        for i in 0..calm_ticks {
            det.observe(t, 0.40 + 0.0001 * wiggle(i), 80_000.0).unwrap();
            t += dt;
        }
        det.observe(t - dt + gap, 0.40 + jump, 80_000.0).unwrap()
    }

    #[test]
    fn detects_shocks_at_any_sampling_cadence_and_across_gaps() {
        // Previously required span >= 0.8 * window after dropping points older than the
        // window, so 180 s sampling and gaps longer than the window were blind.
        for dt in [10.0, 60.0, 120.0, 150.0, 180.0, 200.0, 250.0] {
            let m = multiplier_after_jump(dt, 120, dt, 0.10);
            assert!(m > 10.0, "dt = {dt}: multiplier {m}");
        }
        let m = multiplier_after_jump(10.0, 200, 301.0, 0.10);
        assert!(m > 10.0, "gap of window + 1: multiplier {m}");
        let m = multiplier_after_jump(10.0, 200, 3600.0, 0.10);
        assert!(m > 10.0, "one-hour gap: multiplier {m}");
    }

    #[test]
    fn sparse_calm_feed_is_not_flagged() {
        for dt in [60.0, 180.0, 301.0] {
            let mut det = RegimeDetector::new(RegimeConfig::default()).unwrap();
            let mut t = 0.0;
            for i in 0..400 {
                det.observe(
                    t,
                    0.40 + 0.0002 * wiggle(i),
                    80_000.0 * (1.0 + 1e-5 * wiggle(i + 3)),
                )
                .unwrap();
                t += dt;
            }
            assert_eq!(det.multiplier(), 1.0, "dt = {dt}");
        }
    }

    #[test]
    fn flat_feed_does_not_collapse_the_baseline() {
        // 72 hours of exactly constant ATM vol, then ordinary ticks.
        let cfg = RegimeConfig::default();
        let floor = (0.5 * cfg.initial_change_sd).powi(2);
        let mut det = RegimeDetector::new(cfg).unwrap();
        let mut t = 0.0;
        for _ in 0..(72 * 360) {
            det.observe(t, 0.40, 80_000.0).unwrap();
            t += 10.0;
        }
        assert!(det.change_var >= floor && det.change_var.is_finite());
        assert_eq!(det.multiplier(), 1.0);
        // A 0.1 vol-point tick after three flat days is ordinary, not a regime shock...
        let m = det.observe(t, 0.401, 80_000.0).unwrap();
        assert!(m < 1.5, "{m}");
        // ...but a tiny tick is certainly not.
        let mut det2 = det.clone();
        assert_eq!(det2.observe(t, 0.4001, 80_000.0).unwrap(), 1.0);
        // And a genuine 3 vol-point jump still triggers.
        let m = det.observe(t + 10.0, 0.43, 80_000.0).unwrap();
        assert!(m > 10.0, "{m}");
    }

    #[test]
    fn forward_shock_alone_raises_multiplier() {
        let mut det = RegimeDetector::new(RegimeConfig::default()).unwrap();
        let mut t = calm(&mut det, 0.0, 1500, 0.40, 80_000.0);
        let mut peak: Real = 1.0;
        for k in 1..=30 {
            // -2% in five minutes at constant ATM vol.
            let f = 80_000.0 * (1.0 - 0.02 * k as Real / 30.0);
            peak = peak.max(det.observe(t, 0.40, f).unwrap());
            t += 10.0;
        }
        assert!(peak > 5.0, "{peak}");
    }

    #[test]
    fn multiplier_is_capped() {
        let cfg = RegimeConfig {
            max_multiplier: 4.0,
            ..RegimeConfig::default()
        };
        let mut det = RegimeDetector::new(cfg).unwrap();
        let mut t = calm(&mut det, 0.0, 200, 0.40, 80_000.0);
        let mut peak: Real = 1.0;
        for k in 1..=30 {
            peak = peak.max(
                det.observe(t, 0.40 + 0.2 * k as Real / 30.0, 80_000.0)
                    .unwrap(),
            );
            t += 10.0;
        }
        assert_eq!(peak, 4.0);
    }

    #[test]
    fn invalid_input_leaves_state_unchanged() {
        let mut det = RegimeDetector::new(RegimeConfig::default()).unwrap();
        det.observe(0.0, 0.4, 100.0).unwrap();
        det.observe(10.0, 0.4, 100.0).unwrap();
        let before = det.clone();
        for (t, v, f) in [
            (5.0, 0.4, 100.0), // time goes backwards
            (f64::NAN, 0.4, 100.0),
            (20.0, 0.0, 100.0),
            (20.0, f64::NAN, 100.0),
            (20.0, 0.4, -1.0),
            (20.0, 0.4, f64::INFINITY),
        ] {
            assert!(det.observe(t, v, f).is_err(), "({t}, {v}, {f})");
        }
        assert_eq!(det.history, before.history);
        assert_eq!(det.last_time, before.last_time);
        assert_eq!(det.multiplier, before.multiplier);
        assert_eq!(det.change_var, before.change_var);
    }

    #[test]
    fn config_is_validated() {
        let ok = RegimeConfig::default();
        assert!(ok.validate().is_ok());
        assert!(RegimeConfig { window: 0.0, ..ok }.validate().is_err());
        assert!(
            RegimeConfig {
                threshold: f64::NAN,
                ..ok
            }
            .validate()
            .is_err()
        );
        assert!(RegimeConfig { decay: -1.0, ..ok }.validate().is_err());
        assert!(
            RegimeConfig {
                max_multiplier: 0.5,
                ..ok
            }
            .validate()
            .is_err()
        );
        assert!(
            RegimeDetector::new(RegimeConfig {
                baseline_tau: 0.0,
                ..ok
            })
            .is_err()
        );
    }
}
