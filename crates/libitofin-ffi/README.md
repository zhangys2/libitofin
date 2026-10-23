# libitofin-ffi

Minimal C ABI for embedding the [`libitofin`](https://crates.io/crates/libitofin) core.

This crate currently exports metadata and stable error-code helpers only
(`libitofin_version`, `libitofin_error_message`). Pricing stays in `libitofin`;
a fuller `cbindgen`-generated header and pricing surface are planned once the
core API settles further.

## Build

```sh
cargo build -p libitofin-ffi
cargo test -p libitofin-ffi
```

## License

[BSD-3-Clause](../../LICENSE) — same as the rest of the workspace.
