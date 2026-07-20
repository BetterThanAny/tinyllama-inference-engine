# AGENTS.md

## Scope

These instructions apply to the entire `tinyllama-inference-engine` project.

This project is a correctness-first C++/CUDA TinyLlama inference and serving engine. Performance claims require reproducible evidence from the designated RTX 3080 Laptop. Never trade numerical validity, memory safety, or benchmark fairness for a larger tokens-per-second number.

## Source of truth

- `PLAN.md` defines the model scope, milestones, acceptance thresholds, benchmark metadata, and risks.
- Update `PLAN.md` when a milestone, benchmark command, supported environment, architectural decision, or installed tool changes.
- Record binary formats, kernel contracts, scheduler decisions, and server protocol choices under `docs/adr/` once available.
- A CUDA milestone is not complete until the real GPU commands have run on the 3080 Laptop.

## Environment boundary

- Mac development may verify formatting, CMake configuration, CPU reference code, tokenizer, loader, documentation, and non-CUDA tests.
- CUDA correctness, Compute Sanitizer, profiling, memory, throughput, TTFT, TPOT, and stability must run on the 16GB RTX 3080 Laptop.
- Never report a CUDA test as passing based on compilation alone, a skipped test, an emulator, or another backend.
- Benchmark reports must state the exact GPU, VRAM, driver, CUDA, power mode, temperature, clock behavior, model checksum, dtype, batch, context, warmup, and sample count.
- If results come from different hardware or software versions, treat them as separate baselines.

## Remote CUDA workflow

- The Mac checkout is the only source of truth. Never edit the WSL mirror or copy source changes back from it.
- Use the Mac SSH alias `my-wsl` by default. Override it only through `TLIE_WSL_HOST` when necessary.
- Before every remote CUDA configure, build, test, sanitizer, profiler, or benchmark command, run `scripts/sync_to_wsl.sh` from the Mac checkout.
- The default WSL mirror is `~/tinyllama-inference-engine`; override it only with a safe home-relative `TLIE_WSL_DIR`.
- The sync script mirrors source with `--delete` only inside that dedicated directory. It excludes `.git/`, Mac build state, virtual environments, models, generated data, and benchmark artifacts so remote-only heavy artifacts survive synchronization.
- Run CUDA commands with `ssh my-wsl 'cd ~/tinyllama-inference-engine && <command>'`. Keep related configure/build/test commands in one quoted remote shell when they depend on the same environment.
- Do not run Git operations or make source edits in the WSL mirror. Do not point the sync script at a pre-existing non-mirror directory.
- Do not install or upgrade the WSL driver, CUDA toolkit, CMake, Python, `mise`, `uv`, Nsight, or any global dependency without explicit user approval.
- If SSH, synchronization, a required tool, the model data, or the real GPU is unavailable, report the CUDA verification as blocked or unverified rather than substituting a Mac result.

## Required workflow

1. Inspect `git status --short`, relevant kernel/runtime code, tests, and the latest benchmark record.
2. Add or update a CPU/CUDA reference test before optimizing a kernel or layout.
3. Verify numerical correctness before measuring speed.
4. Use a microbenchmark to establish the local effect, then run end-to-end prefill/decode benchmarks.
5. Run sanitizer checks for memory-sensitive changes.
6. Compare against the previous fixed baseline and record regressions as well as gains.
7. Report verified results, skipped environments, and non-findings explicitly.

## Correctness invariants

- Model config, tensor shapes, dtype, strides, and weight checksum are validated before execution.
- Tokenizer behavior is tested against a fixed reference corpus.
- Each custom kernel has a reference implementation and documented numerical tolerance.
- Greedy generation uses fixed inputs and must match golden tokens before sampling benchmarks.
- KV Cache ownership, bounds, position mapping, and release behavior are explicit.
- No optimization may silently change model architecture, sequence length, precision, or generated token count in a comparison.
- NaN/Inf detection is enabled in correctness and long-context tests.
- Error paths return structured errors rather than aborting the process when recovery is possible.

## Performance rules

- Separate model load, warmup, prefill, decode, sampling, and transfer time.
- Use CUDA events for GPU timing and a monotonic host clock for service latency.
- Synchronize deliberately; do not add global synchronization merely to simplify timing.
- Use identical model, prompt tokens, output length, sampling parameters, and timing boundaries for cross-engine comparisons.
- Report median and tail/distribution information, not only the best run.
- Check thermal throttling before accepting laptop benchmark results.
- An optimization is a success only if it meets the plan threshold without violating correctness.
- If INT8 or fusion does not improve the measured bottleneck, keep the profiler-backed non-finding instead of claiming speedup.

## CUDA and C++ conventions

- Prefer C++20 with explicit ownership and RAII.
- Avoid hidden global CUDA state.
- Check CUDA and cuBLAS return codes at subsystem boundaries.
- Keep kernel launch configuration and supported shape assumptions documented near the kernel.
- Do not allocate device memory inside the per-token hot path unless the design explicitly requires it and the benchmark justifies it.
- Keep CPU reference code readable even when it is not fast.
- Use CMake presets for CPU debug and CUDA release environments.
- Use sanitizers in CPU debug builds and Compute Sanitizer for selected CUDA tests.

## Scheduler and server invariants

- Every request has explicit queued/running/completed/cancelled/failed state.
- Client cancellation and timeout must release scheduler and KV Cache resources.
- Continuous batching must not mix incompatible model or generation configurations.
- Fairness and starvation behavior must be tested with mixed short and long requests.
- OpenAI-compatible endpoints must return stable, documented error shapes.
- Server metrics must include queue time, TTFT, TPOT, output tok/s, active sequences, and KV utilization.

## Environment and dependencies

- Pin non-driver tools with `mise` where practical and use CMake presets for compiler/CUDA settings.
- Python helper dependencies use `uv add` and `uv run`.
- Do not install any global package, CUDA toolkit, driver, Nsight component, or system dependency without asking the user.
- Never use personal absolute paths in CMake, code, model manifests, or scripts.
- Models and generated benchmark artifacts must not be committed unless their size and license are intentionally approved.
- Secrets for optional remote model comparison use environment variables or `op://...` references.

## Testing and verification

Expected CPU verification when available:

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure
```

Expected GPU verification on the 3080 Laptop when available:

```bash
scripts/sync_to_wsl.sh
ssh my-wsl 'cd ~/tinyllama-inference-engine && \
cmake --preset cuda-release && \
cmake --build --preset cuda-release && \
ctest --preset cuda-release --output-on-failure && \
compute-sanitizer --tool memcheck ./build/cuda-release/tests/kernel_tests && \
python scripts/compare_logits.py --reference data/golden --engine build/cuda-release/tinyllama && \
python scripts/benchmark.py --contexts 128 512 2048 4096 --batches 1 2 4 8 && \
python scripts/openai_smoke_test.py --base-url http://127.0.0.1:8080/v1'
```

Inspect configure and test output for skipped CUDA tests, `0 tests`, fallback backends, unsupported compute capability, and missing model data before accepting a green exit code.

## Privacy and Git hygiene

- Do not hardcode user home paths or personal identities.
- Use placeholder emails in examples and fixtures.
- Never add AI co-author or AI attribution trailers.
- Ask before publishing model artifacts, benchmark results, releases, or container images.
- Ask before destructive cleanup, force-pushes, driver/toolkit changes, or CI permission changes.
- Exclude `.git/`, `.DS_Store`, model weights, and large benchmark caches from shared archives unless explicitly requested.
