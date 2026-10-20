---
title: Flat-spread acceptance bar
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: null
resolution: "The flat-spread finite-difference acceptance bar is the existing QuantLib binomial tolerances plus mesh self-convergence. No absolute published price."
map: ../maps/convertible-bond-equity-credit.md
---

# Flat-spread acceptance bar

## Question

For the flat-spread finite-difference engine, is the acceptance bar the existing QuantLib binomial tolerances plus mesh self-convergence, with no absolute published price?

## Context

[Published numerical checks](cb-published-numerical-checks.md) found no recoverable price in Tsiveriotis–Fernandes (1998). Ayache–Forsyth–Vetzal (2003) print a TF value of about 123.97 at \(S=100\), and they do not print the dividend yield. The hazard-rate table is a separate decision, [Hazard-rate acceptance oracle](cb-hazard-acceptance-oracle.md).

## Resolution

Resolved 2026-10-19 by agreement with the recommendation.

The flat-spread finite-difference acceptance bar is the existing QuantLib binomial tolerances plus mesh self-convergence. No absolute published price is required.
