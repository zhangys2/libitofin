---
title: PDE contract features
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: null
resolution: "First finite-difference engine exercises voluntary conversion, issuer call, holder put, and fixed coupons. Soft call and discrete dividends are outside this cut."
map: ../maps/convertible-bond-equity-credit.md
---

# PDE contract features

## Question

Which of voluntary conversion, issuer call, holder put, soft call, fixed coupons, and discrete dividends must the first finite-difference engine exercise?

## Context

`ConvertibleFixedCouponBond`, `ConvertibleZeroCouponBond`, and `ConvertibleFloatingRateBond` already carry callability (including soft call), coupons, and a dividend schedule into `BinomialConvertibleEngine`. The finite-difference engine needs its own first-cut feature set. This ticket is the user’s decision. It does not decide mesh treatment of a trigger or a dividend drop.

## Resolution

Resolved 2026-10-19 by agreement with the recommendation.

The first finite-difference engine exercises voluntary conversion, issuer call, holder put, and fixed coupons. Soft call and discrete dividends are outside this cut, so their mesh rules are not decided here. Floating-rate coupons were not in the first-cut set.
