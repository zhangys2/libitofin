"""QuantLib 1.44 callablebonds.cpp:1113 testEffectiveDurationAndConvexity values."""
import QuantLib as ql
assert ql.__version__ == "1.44"
settlement = ql.Date(30, 11, 2023)
ql.Settings.instance().evaluationDate = settlement
effective, maturity, first = ql.Date(20, 5, 2021), ql.Date(1, 6, 2029), ql.Date(1, 12, 2021)
cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
dc = ql.Thirty360(ql.Thirty360.ISDA)
schedule = ql.Schedule(effective, maturity, ql.Period(6, ql.Months), cal, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, True, first)
calls = ql.CallabilitySchedule()
for price, d in [(102.438, ql.Date(1, 6, 2024)), (101.219, ql.Date(1, 6, 2025)), (100.0, ql.Date(1, 6, 2026)), (100.0, ql.Date(1, 6, 2029))]:
    calls.append(ql.Callability(ql.BondPrice(price, ql.BondPrice.Clean), ql.Callability.Call, d))
bond = ql.CallableFixedRateBond(2, 100.0, schedule, [0.04875], dc, ql.Unadjusted, 100.0, effective, calls)
flat = ql.YieldTermStructureHandle(ql.FlatForward(2, cal, 0.05, dc))
hw = ql.HullWhite(flat, 0.03, 0.012)
steps = (maturity - settlement) // 30
bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(hw, steps))
oas = bond.OAS(70.926, flat, dc, ql.Compounded, ql.Semiannual, settlement)
shift = 0.001
dur = bond.effectiveDuration(oas, flat, dc, ql.Compounded, ql.Semiannual, shift)
conv = bond.effectiveConvexity(oas, flat, dc, ql.Compounded, ql.Semiannual, shift)
acc = bond.accruedAmount(settlement)
p = [bond.cleanPriceOAS(s, flat, dc, ql.Compounded, ql.Semiannual, settlement) + acc for s in (oas, oas + shift, oas - shift)]
exp_dur = (p[2] - p[1]) / (2 * p[0] * shift)
exp_conv = (p[2] + p[1] - 2 * p[0]) / (p[0] * shift * shift)
wrong = (p[2] - p[1]) / (2 * (p[0] - acc) * shift)
for k, v in dict(steps=steps, settle=bond.settlementDate().ISO(), oas=oas, dur=dur, conv=conv, acc=acc, acc_default=bond.accruedAmount(), P0=p[0], exp_dur=exp_dur, exp_conv=exp_conv, wrong=wrong, reldur=abs(dur/exp_dur-1), relconv=abs(conv/exp_conv-1)).items():
    print(k, repr(v))
for d in (-1e-10, 1e-10):
    print("shift", d, bond.effectiveDuration(oas + d, flat, dc, ql.Compounded, ql.Semiannual, shift) - dur, bond.effectiveConvexity(oas + d, flat, dc, ql.Compounded, ql.Semiannual, shift) - conv)
