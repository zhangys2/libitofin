using Xunit;

namespace Itofin.Tests;

public class PricingTests
{
    // Haug rows from libitofin europeanoption testValues, Actual/360, 15 Jun 2026.
    [Fact]
    public void EuropeanPricesMatchTheRustOracle()
    {
        using var session = Session.Open();
        using var settings = session.NewSettings();
        var today = Date.Create(15, 6, 2026);
        settings.SetEvaluationDate(today);
        using var dc = session.Actual360();
        Assert.Equal("Actual/360", dc.Name());
        Near(Price(session, settings, dc, today, OptionType.Call, 65, 60, 0, 0.08, 0.30, 90), 2.1333684449161985);
        Near(Price(session, settings, dc, today, OptionType.Put, 95, 100, 0.05, 0.10, 0.20, 180), 2.4647876467558127);
        var error = Assert.Throws<ItofinException>(() => session.NewEuropean((OptionType)99, 65, today, settings));
        Assert.Equal(1, error.Code);
    }

    [Fact]
    public void BlackImpliedVolatilityRecoversTheRustPrice()
    {
        // black_formula(Call, 100, 100, stddev 0.2, discount 1, displacement 0).
        const double price = 7.965567455405798;
        var std = BlackFormula.ImpliedStdDev(BlackFormula.Call, 100, 100, price, accuracy: 1e-12);
        var seeded = BlackFormula.ImpliedStdDev(BlackFormula.Call, 100, 100, price, guess: 0.1, accuracy: 1e-12);
        var vol = BlackFormula.ImpliedVolatility(BlackFormula.Call, 100, 100, 1, price, accuracy: 1e-12);
        Near(std, 0.2, 1e-8);
        Near(seeded, std, 1e-12);
        Near(vol, 0.2, 1e-8);
        Assert.Throws<ItofinException>(() => BlackFormula.ImpliedStdDev(BlackFormula.Call, 100, 100, -0.1));
    }

    [Fact]
    public void BermudanExerciseDatesRoundTripSorted()
    {
        using var session = Session.Open();
        var april = Date.Create(15, 4, 2025).Serial;
        var july = Date.Create(15, 7, 2025).Serial;
        var january = Date.Create(15, 1, 2026).Serial;
        using var exercise = session.NewBermudanExercise([july, april, january]);
        Assert.Equal(new[] { april, july, january }, exercise.Dates());
    }

    [Fact]
    public async Task DisposeOnAnotherThreadDropsTheContext()
    {
        var session = Session.Open();
        var dc = session.Actual360();
        await Task.Run(() => session.Dispose());
        Assert.Throws<ObjectDisposedException>(() => dc.Name());
        dc.Dispose();
    }

    [Fact]
    public void LibraryReportsAbiVersionOne()
    {
        Assert.Equal(1u, ItofinLibrary.AbiVersion);
        Assert.False(string.IsNullOrEmpty(ItofinLibrary.Version));
    }

    static double Price(Session session, Settings settings, DayCounter dc, Date today, OptionType kind, double strike, double spot, double dividend, double riskFree, double vol, long days)
    {
        using var process = session.NewBlackScholes(spot, riskFree, dividend, vol, today, dc);
        using var option = session.NewEuropean(kind, strike, today.AddDays(days), settings);
        option.SetBlackScholesEngine(process);
        var npv = option.Npv();
        Assert.False(double.IsNaN(option.Delta()));
        return npv;
    }

    static void Near(double got, double want, double tol = 1e-10) =>
        Assert.True(Math.Abs(got - want) <= tol, $"got {got:R} want {want:R}");
}
