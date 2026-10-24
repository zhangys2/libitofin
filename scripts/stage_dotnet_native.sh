#!/bin/sh
# Copy the release cdylib into the NuGet runtime layout for this machine.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$root"
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) rid=linux-x64; lib=libitofin_ffi.so ;;
  Linux-aarch64) rid=linux-arm64; lib=libitofin_ffi.so ;;
  Darwin-arm64) rid=osx-arm64; lib=libitofin_ffi.dylib ;;
  Darwin-x86_64) rid=osx-x64; lib=libitofin_ffi.dylib ;;
  *) echo "unsupported platform $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
src="target/release/$lib"
test -f "$src"
rel="runtimes/$rid/native/$lib"
mkdir -p "sdk/dotnet/Itofin/$(dirname "$rel")"
cp "$src" "sdk/dotnet/Itofin/$rel"
echo "$rel"
