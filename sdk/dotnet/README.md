# .NET bindings

C# bindings for libitofin. They call the checked-in C ABI
(`crates/libitofin-ffi/include/itofin.h`), the same surface the Go SDK uses.
There is no second Rust binding crate.

This scaffold covers a thread-confined `Session` (`SafeHandle` for the native
context and for object ids), European Black-Scholes prices, Black implied
volatility, day-counter names, and Bermudan exercise dates. A context belongs
to the thread that created it. `Dispose` posts destruction back to that thread.
Dispose sessions and objects explicitly.

The package targets `net8.0`. Native libraries belong under
`runtimes/<rid>/native/` (`libitofin_ffi.so`, `libitofin_ffi.dylib`, or
`itofin_ffi.dll`). The loader also honors `ITOFIN_NATIVE_PATH`.

## Build and test

Requires the .NET 8 SDK and the repo Rust toolchain.

```sh
cargo build -p libitofin-ffi --release
bash scripts/stage_dotnet_native.sh
dotnet test sdk/dotnet/Itofin.Tests
dotnet pack sdk/dotnet/Itofin/Itofin.csproj -o dist/dotnet
```

`stage_dotnet_native.sh` fills `linux-x64`, `linux-arm64`, `osx-arm64`, or
`osx-x64` for the machine it runs on. A release package would merge those
folders. Windows (`win-x64`) is recognized by the loader and not built here.

## Regenerate P/Invoke declarations

`scripts/gen_dotnet_bindings.py` reads the C header and rewrites
`sdk/dotnet/Itofin/Native/NativeMethods.g.cs` for an allowlist in that script.
CI fails if the committed file drifts. Extend the allowlist when wrapping more
of the header; do not edit the generated file.

```sh
python3 scripts/gen_dotnet_bindings.py
```

## Use

```csharp
using var session = Session.Open();
var today = Date.Create(15, 6, 2026);
using var settings = session.NewSettings();
settings.SetEvaluationDate(today);
using var dc = session.Actual360();
using var process = session.NewBlackScholes(60, 0.08, 0, 0.30, today, dc);
using var option = session.NewEuropean(OptionType.Call, 65, today.AddDays(90), settings);
option.SetBlackScholesEngine(process);
var npv = option.Npv();
```

Black implied volatility is stateless. Its option code is `BlackFormula.Call`
(1) or `BlackFormula.Put` (-1), which is not `OptionType`.
