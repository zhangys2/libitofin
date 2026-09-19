# AGENTS.md

## Cursor Cloud specific instructions

`libitofin` is a Rust Cargo workspace (a QuantLib port; pinned Rust 1.96.0 via
`rust-toolchain.toml`) plus a PyO3/maturin Python-bindings crate `itofin`
(`crates/itofin-py`, CPython 3.10+ via `abi3-py310`; 3.13 is the primary CI
target). It is a library — there are no runtime services, ports, or databases.
Standard dev commands live in the README "Getting started (development)" section
and in `.github/workflows/pre-commit.yml` (workflow name `pull-request`).

### Python 3.10+ is required for workspace-level cargo commands
`crates/itofin-py` pins PyO3's `abi3-py310`, so anything that builds the whole
workspace needs a Python **≥ 3.10** interpreter visible to PyO3. Prefer the
repo venv (3.13) so local builds match CI's primary target:
- Activate the prebuilt venv first: `source /workspace/.venv/bin/activate`, then
  run `cargo build --workspace` / `cargo test --workspace --locked`. The venv
  supplies Python 3.13 that PyO3 discovers automatically.
- Or scope to the core crate (no Python needed): `cargo build -p libitofin`,
  `cargo test -p libitofin`.

### The environment (baked into the VM snapshot)
- A uv-managed Python 3.13 is installed; `uv` is on `PATH` (via `~/.bashrc`).
- A gitignored repo-local venv at `/workspace/.venv` has `maturin`, `pytest`,
  and `pip` installed. Activate it for workspace cargo commands and all Python work.

### Python bindings workflow
From an activated venv: `maturin develop -m crates/itofin-py/Cargo.toml` builds
and installs the `itofin` extension, then `pytest crates/itofin-py/tests` runs
the oracle. Notes:
- `maturin develop` requires `pip` inside the venv (already present). If the venv
  is ever recreated with `uv venv`, either `uv pip install pip` or use
  `maturin develop --uv`.
- After editing any binding source under `crates/itofin-py/src`, re-run
  `maturin develop` — a running/imported `itofin` will not pick up changes.

### Lint gate nuance
CI runs `pre-commit` only on files changed in the PR (see the `--from-ref/--to-ref`
step in the workflow), not the whole tree. A whole-repo `cargo fmt --all --check`
reports pre-existing formatting drift that CI does not gate — don't treat it as a
regression. `cargo clippy --all-targets` is clean.
