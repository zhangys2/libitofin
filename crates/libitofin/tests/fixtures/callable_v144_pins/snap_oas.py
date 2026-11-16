import QuantLib as ql

assert ql.__version__ == "1.44", ql.__version__

today = ql.Date(18, 5, 2021)
ql.Settings.instance().evaluationDate = today
cal = ql.UnitedStates(ql.UnitedStates.FederalReserve)
dc = ql.Thirty360(ql.Thirty360.USA)
ts = ql.RelinkableYieldTermStructureHandle()
ts.linkTo(ql.FlatForward(today, 0.02, ql.Actual365Fixed()))
issue = ql.Date(20, 5, 2021) - 2 * 366
sched = ql.MakeSchedule(
    issue,
    ql.Date(14, 2, 2026),
    ql.Period(ql.Semiannual),
    calendar=cal,
    convention=ql.Unadjusted,
    terminalDateConvention=ql.Unadjusted,
    backwards=True,
    endOfMonth=False,
)


def bond(call_date):
    calls = ql.CallabilitySchedule(
        [ql.Callability(ql.BondPrice(100.0, ql.BondPrice.Clean), ql.Callability.Call, call_date)]
    )
    b = ql.CallableFixedRateBond(
        2, 100.0, sched, [0.05] * (len(sched) - 1), dc, ql.Following, 100.0, issue, calls
    )
    b.setPricingEngine(ql.TreeCallableFixedRateBondEngine(ql.HullWhite(ts, 1e-12, 0.003), 40))
    return b


print("callablebonds.cpp:571 OAS(cleanPrice() - 2) per business-day call date")
for i in range(-10, 11):
    cd = ql.Date(14, 2, 2022) + i
    if cal.isBusinessDay(cd):
        b = bond(cd)
        print(cd.ISO(), repr(b.OAS(b.cleanPrice() - 2.0, ts, dc, ql.Continuous, ql.Semiannual)))

print("settlement-normalized cleanPriceOAS (Rust quote convention)")
for d in (4, 7, 11):
    b = bond(ql.Date(d, 2, 2022))
    s = b.settlementDate()
    acc = b.accruedAmount(s)
    disc = ts.discount(s)
    for spread in (0.0, 0.01):
        raw = b.cleanPriceOAS(spread, ts, dc, ql.Continuous, ql.Semiannual, s)
        print(d, spread, repr((raw + acc) / disc - acc))
