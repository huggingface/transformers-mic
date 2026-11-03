# Changelog

All notable changes to `transformers-mic` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-11-03

### Added

- The CLI guide opens with a section for coding agents: a copy-paste prompt for a local `tmic run --mode agent`,
  the exit codes, the answer format and who decides each question kind (#19).
- `tmic shrink` shows the model's own Transformers tester config next to the fixture (#160).

### Changed

- Planner: the first draft is greedy (temperature 0), and its example is the closest approved contract in the
  fleets (by architecture, `model_type`, sub-configs, family, input modality and head) instead of BERT. It also
  receives one real parameter-name pattern per layer kind (#159).
- Shrink takes the model tester's widths for size fields the rule left at the checkpoint's value, and rebuilds
  with the rule's sizes when the tester's fail to build (#160, #161).
- Chat templates see a fixed date while recording and checking, and the date is stored with each baseline, so
  templates that insert today's date no longer change baselines from day to day (#166).

### Fixed

- Cloud baselines refuse a CPU host without native bf16/fp16 before measuring and cancel the run. A collection
  already measured on the wrong CPU class is set aside, so a retry dispatches again (#162, #163).
- SentencePiece-only tokenizers are detected when `sentencepiece` is not installed (#144).

## [0.1.0] - 2026-11-02

First release on PyPI. `tmic` is alpha: generated contracts still need code review, and the cloud baseline and
publication paths are for maintainers of the `huggingface-mic` organization.

### Added

- `tmic run`: one command from a real checkpoint to a validated MIC. It drafts or reuses a contract, builds the
  slim fixture, records and reviews baselines, checks every declared variant and, for maintainers, publishes the
  model and updates the fleet. `--local-only` measures on your machine and uploads nothing. Human mode asks in the
  terminal; `--mode agent` speaks a JSON question/answer protocol. Runs resume from `.mic/automation.json`.
- Step-by-step commands: `create`, `update`, `build`, `record`, `review`, `check`, `status`, `replay`,
  `publish`, `export`, `pin`, `retire` and `batch`.
- Contract drafting through Hugging Face Inference Providers (default `deepseek-ai/DeepSeek-V4.1-Flash-1017` on
  Together), billed to the `huggingface` organization unless `--bill-to` says otherwise. Drafts that enable
  `trust_remote_code`, import names Transformers does not export, or train on `no_grad` outputs are rejected.
  Bounded contract repair proposes a fix for a failing draft and applies it only after approval.
- The fixture builder: shrinks the real config while keeping every layer kind the checkpoint builds, copies
  processor files byte for byte, keeps the checkpoint's tensor names and file format, keeps sources of 20M
  parameters or fewer whole, and refuses test checkpoints as sources. Weights are random when they pass the
  input-sensitivity gate and overfit on the contract's batch otherwise.
- `tmic shrink`: a page comparing the fixture's layer stacks, parameters and config with its checkpoint.
- Baselines per platform (`cpu-arm64`, `cpu-x86_64`, `cpu-x86_64-bf16`, CUDA) and variant. `review --approve`
  accepts only new and unchanged values; changed values need a person.
- The fleet as CSV tables in the `huggingface-mic/transformers-fleet` dataset, converted to and from the JSON
  lock the runners read. Transformers pins a dataset commit in `tests/contracts.lock`.
- Cloud baseline collections on CPU and CUDA runners (`tmic baselines`), through the dashboard or directly
  through GitHub Actions.
- `tmic auth` for the Transformers CI dashboard.
