# Rust Workspace CI Design

**Goal:** Make pull requests validate the complete Rust workspace in addition
to the existing pre-commit and Python binding checks.

## Design

Add a `rust-workspace` job to `.github/workflows/pull-request.yml`. The job
runs on `ubuntu-latest`, checks out the repository, installs the pinned Rust
1.96.0 toolchain, runs `cargo test --workspace --locked`, and then runs
`cargo build --workspace --locked`.

Tests and build remain separate steps so failures identify whether the
workspace is behaviorally broken or only fails to compile. No new actions,
dependency caches, clippy policy, or platform matrix are introduced.

## Validation

Validate YAML structure with the existing pre-commit configuration where
available, run the same Cargo commands locally, and inspect the workflow diff.
The change is limited to pull-request and manual workflow execution.
