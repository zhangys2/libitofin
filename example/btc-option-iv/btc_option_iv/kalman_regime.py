"""External volatility-regime detector for the Kalman smile filter.

Mirrors ``kalman_regime.rs`` in the Rust library. The detector watches the ATM
vol and the forward and returns a process-noise time multiplier: the filter
should advance its covariance by ``dt * multiplier`` so it loosens during shocks
and tightens again as the multiplier decays. Use a threshold of 4 when the ATM
vol is derived from quotes instead of the exchange mark. Do not combine it with
``FilterConfig.adapt_rate``: both would react to the same shock.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

SECONDS_PER_YEAR = 365.0 * 86_400.0
BASELINE_FLOOR_FRACTION = 0.5


@dataclass(frozen=True)
class RegimeConfig:
    window: float = 300.0
    threshold: float = 3.0
    decay: float = 1200.0
    max_multiplier: float = 100.0
    baseline_tau: float = 14_400.0
    initial_change_sd: float = 0.001

    def __post_init__(self):
        for name in (
            "window",
            "threshold",
            "decay",
            "baseline_tau",
            "initial_change_sd",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.max_multiplier) or self.max_multiplier < 1:
            raise ValueError("max_multiplier must be finite and at least 1")


class RegimeDetector:
    def __init__(self, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()
        self._history: deque[tuple[float, float, float]] = deque()
        self._change_var = self.config.initial_change_sd**2
        self._multiplier = 1.0
        self._last_time: float | None = None

    @property
    def multiplier(self) -> float:
        return self._multiplier

    def observe(self, time: float, atm_vol: float, forward: float) -> float:
        """Feed one observation; invalid input raises and leaves the state unchanged."""
        if not math.isfinite(time):
            raise ValueError("time must be finite")
        if not math.isfinite(atm_vol) or atm_vol <= 0:
            raise ValueError("ATM volatility must be finite and positive")
        if not math.isfinite(forward) or forward <= 0:
            raise ValueError("forward must be finite and positive")
        if self._last_time is not None and time < self._last_time:
            raise ValueError("observation times must be non-decreasing")

        cfg = self.config
        dt = 0.0 if self._last_time is None else time - self._last_time
        self._last_time = time
        ln_forward = math.log(forward)
        history = self._history
        history.append((time, atm_vol, ln_forward))
        # Keep one sample at or before the window edge so sparse feeds and gaps
        # still have a reference to compare against.
        while len(history) > 2 and time - history[0][0] > cfg.window:
            history.popleft()
        t0, vol0, lnf0 = history[0]
        span = time - t0

        z = 0.0
        dvol = 0.0
        measured = span > 0
        span_ratio = span / cfg.window
        if measured:
            dvol = atm_vol - vol0
            forward_sd = atm_vol * math.sqrt(span / SECONDS_PER_YEAR)
            z_forward = abs((ln_forward - lnf0) / forward_sd)
            z_vol = abs(dvol) / math.sqrt(self._change_var * span_ratio)
            z = max(z_forward, z_vol)

        target = (
            min((z / cfg.threshold) ** 2, cfg.max_multiplier)
            if z > cfg.threshold
            else 1.0
        )
        self._multiplier = max(
            target, 1.0 + (self._multiplier - 1.0) * math.exp(-dt / cfg.decay)
        )

        if measured and z <= cfg.threshold:
            a = min(dt / cfg.baseline_tau, 1.0)
            floor = (BASELINE_FLOOR_FRACTION * cfg.initial_change_sd) ** 2
            self._change_var = max(
                (1.0 - a) * self._change_var + a * dvol * dvol / span_ratio, floor
            )
        return self._multiplier
