---
title: Cash-gamma scalping identity
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: null
resolution: "Caller supplies the interval (Δt, S′). No-default wealth P&L is grid revaluation minus Δ shares, plus coupon and financing. Continuation split is ½Γ(ΔS)² + Θ_B Δt. Credit sits inside grid theta. Whether a realized default jump is returned stays open."
map: ../maps/convertible-bond-equity-credit.md
---

# Cash-gamma scalping identity

## Question

What discrete cash-gamma P&L identity does convertible arbitrage use, and which of its terms are determined by the solved convertible grid?

## Context

The destination includes grid Greeks and a discrete hedge P&L identity (rehedge delta, earn gamma, pay theta and credit). It excludes order routing. This ticket names the identity and which terms the PDE solution fixes. The caller supplies the interval. Whether a realized default jump is returned is [Default-jump term in the gamma identity](cb-gamma-default-jump.md).

## Resolution

Resolved 2026-10-19 by [Cash-gamma scalping identity](2de811a7-c669-49ef-8466-f1a8938a72d7).

The caller supplies \(\Delta t\) and the new spot \(S'\). There is no embedded rehedge clock. With \(\Delta=\partial V/\partial S\), \(\Gamma=\partial^2 V/\partial S^2\), and \(\Theta=\partial V/\partial t\) from the total-value grid, the no-default wealth change is

\[
\Delta\Pi = \bigl[V(S',t+\Delta t)-V(S,t)\bigr] - \Delta(S'-S) + C_{[t,t+\Delta t]} + (r_{\mathrm{rebate}}-q)\Delta S\,\Delta t - r_{\mathrm{fund}} V\,\Delta t.
\]

Its continuation split is \(\tfrac12\Gamma(\Delta S)^2 + \Theta_B\Delta t + C + \rho\), where \(\Theta_B = \Theta - r_{\mathrm{fund}} V + (r_{\mathrm{rebate}}-q)S\Delta\). On a Tsiveriotis–Fernandes grid the credit carry sits inside \(V\) and \(\Theta\) through the cash component \(C\). Adding it again double-counts. A realized default jump replaces the diffusion scalp for that interval; whether the library returns it is a separate decision.
