"""Shared accessors for a snapshot's total variance cubic reference smile."""

from __future__ import annotations

from itofin import ItofinError

_REFERENCE_ERRORS = (ItofinError, ValueError, OverflowError)


def reference_vol(view, strike: float) -> float | None:
    """Reference IV at ``strike``, or None if there is no usable reference."""
    section = view.reference_section
    if section is None:
        return None
    try:
        return float(section.volatility(strike))
    except _REFERENCE_ERRORS:
        return None


def butterfly_report(view):
    """Durrleman butterfly report for the reference smile, or None."""
    section = view.reference_section
    if section is None:
        return None
    try:
        return section.butterfly_report
    except _REFERENCE_ERRORS:
        return None


def butterfly_markdown(report, *, label: str = "Total variance smile") -> str:
    """One-line markdown summary of a butterfly report ('' when absent)."""
    if report is None:
        return ""
    status = "⚠️ Arbitrage detected" if report.has_arbitrage else "✅ Arbitrage-free"
    return (
        f"**{label}:** {status} (min g(k) = {report.min_density:.4f}, "
        f"final λ = {report.final_smoothing}, ramps = {report.ramp_iterations})"
    )
