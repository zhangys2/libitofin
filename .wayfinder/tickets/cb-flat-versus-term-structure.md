---
title: Flat versus term-structure coefficients
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: null
resolution: "The finite-difference engine freezes flat r, q, and σ at the conversion exercise date, matching BinomialConvertibleEngine."
map: ../maps/convertible-bond-equity-credit.md
---

# Flat versus term-structure coefficients

## Question

Does the finite-difference engine freeze flat \(r\), \(q\), and \(\sigma\) at the conversion exercise date, as `BinomialConvertibleEngine` does, or read the full `GeneralizedBlackScholesProcess` term structures?

## Context

The binomial engine samples flat \(r\), \(q\), and \(\sigma\) from the process at the conversion exercise date, then builds a constant-coefficient Cox–Ross–Rubinstein lattice. Matching that freeze keeps the flat-spread oracle comparable. Reading the term structures changes the comparison. This ticket is the user’s decision.

## Resolution

Resolved 2026-10-19 by agreement with the recommendation.

The finite-difference engine freezes flat \(r\), \(q\), and \(\sigma\) at the conversion exercise date, as `BinomialConvertibleEngine` does.
