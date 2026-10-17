using System.Collections.Concurrent;
using System.Reflection;
using Microsoft.Win32.SafeHandles;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Text;
using Itofin.Native;

namespace Itofin;

// SafeHandle ownership over the C ABI. ReleaseHandle runs on the context's
// owner thread; libitofin rejects cross-thread use and destruction.

public sealed class ItofinException(int code, string message) : Exception($"itofin ({code}): {message}")
{
    public int Code { get; } = code;
}

public enum OptionType { Call = 0, Put = 1 }

public readonly record struct Date(int Serial)
{
    public static Date Create(int day, int month, int year) => new(Calls.DateNew(day, month, year));
    public Date AddDays(long days) => new(Calls.DateShift(Serial, days));
}

public static class ItofinLibrary
{
    public static uint AbiVersion => NativeMethods.itofin_abi_version();
    public static string Version => Calls.Version();
}

public static class BlackFormula
{
    public const int Call = 1;
    public const int Put = -1;

    public static double ImpliedStdDev(int optionType, double strike, double forward, double blackPrice, double discount = 1, double displacement = 0, double? guess = null, double accuracy = 1e-12, long maxIterations = 100) =>
        Calls.ImpliedStdDev(optionType, strike, forward, blackPrice, discount, displacement, guess, accuracy, maxIterations);

    public static double ImpliedVolatility(int optionType, double strike, double forward, double expiry, double blackPrice, double discount = 1, double displacement = 0, double accuracy = 1e-12, long maxIterations = 100) =>
        Calls.ImpliedVol(optionType, strike, forward, expiry, blackPrice, discount, displacement, accuracy, maxIterations);
}

public sealed class Session : IDisposable
{
    readonly NativeGate _gate;
    readonly ContextHandle _context;

    Session(NativeGate gate, ContextHandle context)
    {
        _gate = gate;
        _context = context;
    }

    public static Session Open()
    {
        if (ItofinLibrary.AbiVersion != 1) throw new ItofinException(1, "incompatible native ABI version");
        var gate = new NativeGate();
        var context = new ContextHandle(gate);
        try
        {
            gate.Run(() =>
            {
                var ptr = Calls.ContextNew();
                context.Reset(ptr);
                gate.Context = ptr;
            });
        }
        catch
        {
            if (context.IsInvalid) gate.Shutdown();
            else context.Dispose();
            throw;
        }
        return new Session(gate, context);
    }

    internal NativeGate Gate => _gate;

    public void Dispose()
    {
        if (_context.IsClosed) return;
        if (!_context.IsInvalid && !_gate.Freed)
        {
            _gate.BeginClose();
            _gate.RunUnchecked(() =>
            {
                if (_gate.Freed) return;
                Calls.ContextFree(_gate.Context);
                _gate.MarkFreed();
            });
        }
        _context.Dispose();
    }

    internal void Call(Action<nint> body) => _gate.Run(() => body(_gate.Context));

    public Settings NewSettings() => new(this, Alloc(Calls.SettingsNew));
    public DayCounter Actual360() => new(this, Alloc(ctx => Calls.DayCounter(ctx, 0)));

    public BlackScholesProcess NewBlackScholes(double spot, double riskFree, double dividend, double volatility, Date reference, DayCounter dayCounter)
    {
        Same(dayCounter);
        return new(this, Alloc(ctx => Calls.BlackScholes(ctx, spot, riskFree, dividend, volatility, reference.Serial, dayCounter.Id)));
    }

    public VanillaOption NewEuropean(OptionType kind, double strike, Date expiry, Settings settings)
    {
        Same(settings);
        return new(this, Alloc(ctx => Calls.Option(ctx, (int)kind, strike, expiry.Serial, settings.Id)));
    }

    public BermudanExercise NewBermudanExercise(ReadOnlySpan<int> dates)
    {
        var copy = dates.ToArray();
        return new(this, Alloc(ctx => Calls.BermudanNew(ctx, copy)));
    }

    void Same(NativeObject other)
    {
        if (!ReferenceEquals(other.Owner, this)) throw new ItofinException(2, "objects belong to different sessions");
    }

    ulong Alloc(Func<nint, ulong> create)
    {
        ulong id = 0;
        Call(ctx => id = create(ctx));
        return id;
    }
}

public sealed class Settings(Session session, ulong id) : NativeObject(session, id)
{
    public void SetEvaluationDate(Date date) => Call(ctx => Calls.SetEvaluationDate(ctx, Id, date.Serial));
}

public sealed class DayCounter(Session session, ulong id) : NativeObject(session, id)
{
    public string Name()
    {
        var name = "";
        Call(ctx => name = Calls.TimeName(ctx, Id));
        return name;
    }
}

public sealed class BlackScholesProcess(Session session, ulong id) : NativeObject(session, id);

public sealed class VanillaOption(Session session, ulong id) : NativeObject(session, id)
{
    public void SetBlackScholesEngine(BlackScholesProcess process)
    {
        if (!ReferenceEquals(Owner, process.Owner)) throw new ItofinException(2, "objects belong to different sessions");
        Call(ctx => Calls.SetEngine(ctx, Id, process.Id));
    }

    public double Npv() => Greek(0);
    public double Delta() => Greek(1);

    double Greek(int field)
    {
        double value = 0;
        Call(ctx => value = Calls.OptionValue(ctx, Id, field));
        return value;
    }
}

public sealed class BermudanExercise(Session session, ulong id) : NativeObject(session, id)
{
    public int[] Dates()
    {
        int[] dates = [];
        Call(ctx => dates = Calls.BermudanDates(ctx, Id));
        return dates;
    }
}

public abstract class NativeObject : IDisposable
{
    readonly ObjectHandle _handle;
    private protected NativeObject(Session session, ulong id)
    {
        Owner = session;
        _handle = new ObjectHandle(session.Gate, id);
    }

    internal Session Owner { get; }
    internal ulong Id => _handle.Id;
    private protected void Call(Action<nint> body) => Owner.Call(body);
    public void Dispose() => _handle.Dispose();
}

static class NativeBootstrap
{
#pragma warning disable CA2255 // Library load hook must run before the first P/Invoke.
    [ModuleInitializer]
    internal static void Init() => NativeLibrary.SetDllImportResolver(typeof(NativeMethods).Assembly, Resolve);
#pragma warning restore CA2255

    static nint Resolve(string library, Assembly assembly, DllImportSearchPath? paths)
    {
        if (library != "itofin_ffi") return 0;
        foreach (var candidate in Probe(assembly))
            if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out var handle)) return handle;
        return 0;
    }

    static IEnumerable<string> Probe(Assembly assembly)
    {
        var file = OperatingSystem.IsWindows() ? "itofin_ffi.dll" : OperatingSystem.IsMacOS() ? "libitofin_ffi.dylib" : "libitofin_ffi.so";
        var arch = RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? "arm64" : "x64";
        var os = OperatingSystem.IsWindows() ? "win" : OperatingSystem.IsMacOS() ? "osx" : "linux";
        var env = Environment.GetEnvironmentVariable("ITOFIN_NATIVE_PATH");
        if (!string.IsNullOrEmpty(env)) yield return Directory.Exists(env) ? Path.Combine(env, file) : env;
        foreach (var root in new[] { AppContext.BaseDirectory, Path.GetDirectoryName(assembly.Location) })
        {
            if (string.IsNullOrEmpty(root)) continue;
            yield return Path.Combine(root, "runtimes", os + "-" + arch, "native", file);
            yield return Path.Combine(root, file);
        }
    }
}

sealed class NativeGate
{
    readonly object _sync = new();
    readonly Thread _thread;
    readonly BlockingCollection<Action> _queue = new();
    int _closed;
    int _freed;

    internal nint Context;
    internal bool Freed => Volatile.Read(ref _freed) != 0;
    internal void MarkFreed() => Volatile.Write(ref _freed, 1);

    internal NativeGate()
    {
        using var ready = new ManualResetEventSlim(false);
        _thread = new Thread(() =>
        {
            ready.Set();
            foreach (var work in _queue.GetConsumingEnumerable())
            {
                try { work(); }
                catch { /* The caller wrapper reports the failure. */ }
            }
        })
        { IsBackground = true, Name = "itofin-session" };
        _thread.Start();
        ready.Wait();
    }

    internal void Run(Action work) => Run(work, true);
    internal void RunUnchecked(Action work) => Run(work, false);

    void Run(Action work, bool rejectIfClosed)
    {
        if (Thread.CurrentThread == _thread)
        {
            if (rejectIfClosed && Volatile.Read(ref _closed) != 0) throw new ObjectDisposedException(nameof(Session));
            work();
            return;
        }
        Exception? error = null;
        using var done = new ManualResetEventSlim(false);
        lock (_sync)
        {
            if (rejectIfClosed && _closed != 0) throw new ObjectDisposedException(nameof(Session));
            _queue.Add(() =>
            {
                try { work(); }
                catch (Exception ex) { error = ex; }
                finally { done.Set(); }
            });
        }
        if (!done.Wait(TimeSpan.FromSeconds(30))) throw new TimeoutException("itofin native call timed out");
        if (error is not null) ExceptionDispatchInfo.Capture(error).Throw();
    }

    internal void BeginClose() { lock (_sync) _closed = 1; }

    internal void Shutdown()
    {
        lock (_sync)
        {
            _closed = 1;
            if (!_queue.IsAddingCompleted) _queue.CompleteAdding();
        }
        if (Thread.CurrentThread != _thread) _thread.Join();
    }
}

sealed class ContextHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    readonly NativeGate _gate;
    internal ContextHandle(NativeGate gate) : base(true) => _gate = gate;
    internal void Reset(nint ptr) => SetHandle(ptr);

    protected override bool ReleaseHandle()
    {
        if (Environment.HasShutdownStarted) return true;
        if (_gate.Freed)
        {
            _gate.Shutdown();
            return true;
        }
        var ptr = handle;
        try
        {
            _gate.BeginClose();
            _gate.RunUnchecked(() =>
            {
                if (_gate.Freed) return;
                Calls.ContextFree(ptr);
                _gate.MarkFreed();
            });
            return true;
        }
        catch { return false; }
        finally { _gate.Shutdown(); }
    }
}

sealed class ObjectHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    readonly NativeGate _gate;
    internal ObjectHandle(NativeGate gate, ulong id) : base(true)
    {
        _gate = gate;
        SetHandle(unchecked((nint)id));
    }

    internal ulong Id
    {
        get
        {
            if (IsClosed || IsInvalid) throw new ObjectDisposedException(nameof(NativeObject));
            return unchecked((ulong)DangerousGetHandle());
        }
    }

    protected override bool ReleaseHandle()
    {
        if (Environment.HasShutdownStarted) return true;
        var id = unchecked((ulong)handle);
        try
        {
            _gate.RunUnchecked(() =>
            {
                if (_gate.Freed) return;
                Calls.HandleRelease(_gate.Context, id);
            });
            return true;
        }
        catch { return true; }
    }
}

static class Calls
{
    internal static unsafe void Check(int status, ItofinError* error)
    {
        if (status == 0) return;
        var message = error == null ? "" : Marshal.PtrToStringUTF8((nint)error->Message) ?? "";
        throw new ItofinException(status, message);
    }

    internal static unsafe string Version() => Marshal.PtrToStringUTF8((nint)NativeMethods.itofin_version()) ?? "";

    internal static unsafe nint ContextNew()
    {
        ItofinError e = default; nint ptr;
        Check(NativeMethods.itofin_context_new(&ptr, &e), &e);
        return ptr;
    }

    internal static unsafe void ContextFree(nint ctx)
    {
        ItofinError e = default;
        Check(NativeMethods.itofin_context_free(ctx, &e), &e);
    }

    internal static unsafe void HandleRelease(nint ctx, ulong id)
    {
        ItofinError e = default;
        var status = NativeMethods.itofin_handle_release(ctx, id, &e);
        if (status != 2) Check(status, &e);
    }

    internal static unsafe int DateNew(int day, int month, int year)
    {
        ItofinError e = default; int serial;
        Check(NativeMethods.itofin_date_new(day, month, year, &serial, &e), &e);
        return serial;
    }

    internal static unsafe int DateShift(int serial, long days)
    {
        ItofinError e = default; int shifted;
        Check(NativeMethods.itofin_date_shift(serial, days, &shifted, &e), &e);
        return shifted;
    }

    internal static unsafe ulong SettingsNew(nint ctx)
    {
        ItofinError e = default; ulong id;
        Check(NativeMethods.itofin_settings_new(ctx, &id, &e), &e);
        return id;
    }

    internal static unsafe void SetEvaluationDate(nint ctx, ulong id, int serial)
    {
        ItofinError e = default;
        Check(NativeMethods.itofin_settings_set_evaluation_date(ctx, id, serial, &e), &e);
    }

    internal static unsafe ulong DayCounter(nint ctx, int kind)
    {
        ItofinError e = default; ulong id;
        Check(NativeMethods.itofin_day_counter_new(ctx, kind, &id, &e), &e);
        return id;
    }

    internal static unsafe string TimeName(nint ctx, ulong id)
    {
        ItofinError e = default; nuint n;
        Check(NativeMethods.itofin_time_name(ctx, id, 0, null, 0, &n, &e), &e);
        if (n == 0) return "";
        var buf = new byte[n];
        fixed (byte* p = buf) Check(NativeMethods.itofin_time_name(ctx, id, 0, p, n, &n, &e), &e);
        return Encoding.UTF8.GetString(buf);
    }

    internal static unsafe ulong BlackScholes(nint ctx, double spot, double riskFree, double dividend, double volatility, int reference, ulong dc)
    {
        ItofinError e = default; ulong id;
        Check(NativeMethods.itofin_black_scholes_new(ctx, spot, riskFree, dividend, volatility, reference, dc, &id, &e), &e);
        return id;
    }

    internal static unsafe ulong Option(nint ctx, int kind, double strike, int expiry, ulong settings)
    {
        ItofinError e = default; ulong id;
        Check(NativeMethods.itofin_option_new(ctx, kind, strike, 0, expiry, 0, settings, &id, &e), &e);
        return id;
    }

    internal static unsafe void SetEngine(nint ctx, ulong option, ulong process)
    {
        ItofinError e = default;
        Check(NativeMethods.itofin_option_set_engine(ctx, option, process, 0, 0, &e), &e);
    }

    internal static unsafe double OptionValue(nint ctx, ulong option, int field)
    {
        ItofinError e = default; double value;
        Check(NativeMethods.itofin_option_value(ctx, option, field, &value, &e), &e);
        return value;
    }

    internal static unsafe double ImpliedStdDev(int optionType, double strike, double forward, double blackPrice, double discount, double displacement, double? guess, double accuracy, long maxIterations)
    {
        ItofinError e = default; double value; double seed = guess ?? 0;
        double* seedPtr = guess.HasValue ? &seed : null;
        Check(NativeMethods.itofin_black_formula_implied_std_dev(optionType, strike, forward, blackPrice, discount, displacement, seedPtr, accuracy, maxIterations, &value, &e), &e);
        return value;
    }

    internal static unsafe double ImpliedVol(int optionType, double strike, double forward, double expiry, double blackPrice, double discount, double displacement, double accuracy, long maxIterations)
    {
        ItofinError e = default; double value;
        Check(NativeMethods.itofin_black_formula_implied_volatility(optionType, strike, forward, expiry, blackPrice, discount, displacement, accuracy, maxIterations, &value, &e), &e);
        return value;
    }

    internal static unsafe ulong BermudanNew(nint ctx, int[] dates)
    {
        ItofinError e = default; ulong id;
        fixed (int* p = dates) Check(NativeMethods.itofin_bermudan_exercise_new(ctx, p, (nuint)dates.Length, &id, &e), &e);
        return id;
    }

    internal static unsafe int[] BermudanDates(nint ctx, ulong id)
    {
        ItofinError e = default; nuint n;
        Check(NativeMethods.itofin_bermudan_exercise_dates(ctx, id, null, 0, &n, &e), &e);
        var dates = new int[n];
        if (n == 0) return dates;
        fixed (int* p = dates) Check(NativeMethods.itofin_bermudan_exercise_dates(ctx, id, p, n, &n, &e), &e);
        return dates;
    }
}
