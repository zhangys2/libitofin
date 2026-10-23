# Contributing to Lib-Itô-Fin

Thanks for helping port QuantLib into idiomatic Rust. The single correctness
rule: **QuantLib's `test-suite/*.cpp` is the oracle** — a feature is done only
when the matching tests are ported and Rust matches the C++ numbers within
tolerance.

## Prerequisites

- Rust toolchain from [`rust-toolchain.toml`](rust-toolchain.toml) (currently
  1.96.0 with `rustfmt` and `clippy`). Plain `cargo` picks it up.
- Python **≥ 3.13** if you touch [`crates/itofin-py`](crates/itofin-py).
- Optional: a local QuantLib checkout symlinked as `QuantLib/` for reference:

  ```sh
  ln -s /path/to/QuantLib QuantLib
  ```

## Build and test

```sh
# Rust workspace
cargo build --workspace
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings

# Quality gates (also run in CI)
pre-commit install
pre-commit run --all-files

# Python bindings
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
maturin develop -m crates/itofin-py/Cargo.toml
pytest crates/itofin-py/tests -v
```

See the root [README](README.md#getting-started-development) for the same flows
with more context.

## Pull requests

- **Small PRs** — aim for ≤350 LOC changed; 400 is a hard cap. Split large
  source ports across tickets.
- **Bottom-up** — never port a module before its lower-layer dependencies.
- **Oracle tests travel with the port** — add or extend Rust/`pytest` coverage
  that asserts QuantLib numbers; do not ship untested pricing surface.
- **Conventional commits** — subject must match:

  ```
  <type>(scope): summary
  ```

  Allowed types include `feat`, `fix`, `docs`, `test`, `refactor`, `chore`,
  `ci`, `build`, `perf`, `style`, `revert`, `hotfix`, `ops`, `incompat`.
  Pre-commit's commit-msg hook enforces this.

## Design decisions that are settled

Do not reopen these in review unless there is a concrete correctness bug. The
full list lives in [`.github/copilot-instructions.md`](.github/copilot-instructions.md);
highlights:

- Observer/handle model uses `Rc` / `RefCell` aliases (`Shared`, `SharedMut`).
- Errors are `QlResult` via `fail!` / `require!` — not panics or `anyhow`.
- `Settings` is an explicit value, not a process-global singleton.
- Core stays single-threaded-mutable; no `async` in the core. `rayon`
  snapshot-and-fan-out is the planned parallelism model, not yet wired in.
- FFI lives in sibling crates; `libitofin` stays FFI-agnostic.

## History provenance

Commits up to and including `0dd6af4` (2026-07-29) were imported from the
upstream `bitbrew` itofin repository. A `(#N)` reference in those subjects
points at **upstream** PR N, not at a `zhangys2/libitofin` PR — this repo's PR
counter restarted at #1 on 2026-08-02. From `1ef83f3` (2026-07-30) onward,
`(#N)` means a PR in this repository.

Note that GitHub renders those old `(#N)` subjects as links into *this* repo;
those links are wrong for anything before the cutoff.

## Security

Please report vulnerabilities privately — see [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions are licensed under the
repository's [BSD-3-Clause](LICENSE) license.
