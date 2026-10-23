# Security Policy

## Supported versions

`libitofin` / `itofin` are pre-1.0 and under active development. Security fixes
are applied on the default branch (`main`) and included in the next release.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security reports.

Prefer one of:

1. **GitHub private vulnerability reporting** — use
   [Security → Advisories → Report a vulnerability](https://github.com/benbenbang/libitofin/security/advisories/new)
   on the canonical upstream repository (or the equivalent page on your fork if
   upstream reporting is unavailable).
2. Email the maintainer listed in package metadata (`bn@bitbrew.dev`) with a
   clear description, impact, and reproduction steps.

You should receive an acknowledgement within a few days. Please give us a
reasonable window to investigate and ship a fix before any public disclosure.

## Scope

In scope: memory unsafety in the Rust core, incorrect FFI contracts that can
corrupt caller memory, and supply-chain issues in published crates/wheels.

Out of scope: theoretical issues that require already-compromised host
environments, or general quantitative model risk / “wrong price” reports that
belong in ordinary bug issues (those should cite the QuantLib oracle case).
