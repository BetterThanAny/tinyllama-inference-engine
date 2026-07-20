# TinyLlama inference engine

[![CPU CI](https://github.com/BetterThanAny/tinyllama-inference-engine/actions/workflows/cpu-ci.yml/badge.svg)](https://github.com/BetterThanAny/tinyllama-inference-engine/actions/workflows/cpu-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> A correctness-first, single-GPU TinyLlama 1.1B inference engine built in C++20 and CUDA. Start
> with the [portfolio overview](docs/portfolio-overview.md) for architecture, verified results, and
> claim boundaries; use [PLAN.md](PLAN.md) for the full audit trail.

## At a glance

| Capability | Current verified scope |
|---|---|
| Numerical paths | CPU FP32 reference, CUDA FP16, hybrid W8A16 weight-only |
| Runtime | Explicit KV cache, prefill/decode, Batch 4, 20 reclaimable request slots |
| Service | Loopback OpenAI-compatible subset with JSON/SSE, timeout, and cancellation |
| GPU verification | RTX 3080 Laptop: 249 kernel checks, clean memcheck and racecheck |
| Measured result | Batch 4 is `2.905x` Batch 1; INT8 memory `-40.434%`, speed non-finding |
| Stability | 1800-second, concurrency-20 run; 3076 terminal requests, 0 failed/leaked requests |

The tracked [RTX 3080 evidence summary](benchmarks/evidence/20260720-rtx3080-m5-summary.md) includes
the fixed workload, thermal qualifications, limitations, and SHA-256 identities of the raw reports.
GitHub Actions intentionally covers model-independent Python contracts and CPU ASan/UBSan tests;
CUDA claims require the designated physical GPU and are not represented by a skipped CI job.

This repository contains the verified M1 CUDA-independent reference, M2 FP16 CUDA runtime, M3
hybrid W8A16 weight-only path, M4 continuous-batching OpenAI-compatible server, and M5 stability and
cross-engine evidence for the pinned TinyLlama-1.1B-Chat-v1.0 model. CUDA milestone acceptance
requires correctness, sanitizer, profiler, and performance commands on the designated 16GB RTX
3080 Laptop.

The Mac checkout is the only source tree. Before every CUDA configure, build, test, sanitizer, or
benchmark command, run `TLIE_WSL_DIR=tinyllama-inference-engine ./scripts/sync_to_wsl.sh` on the
Mac. Run the corresponding command only in `~/tinyllama-inference-engine` through `ssh my-wsl`;
never edit or run Git in that mirror. Model files and generated reports are intentionally excluded
from source synchronization and must be prepared separately from their pinned checksums.

## Requirements

- CMake 3.25 or newer and a C++20 compiler
- `mise` with the Python version pinned by `.mise.toml`
- `uv`
- Approximately 7 GB free for the upstream BF16 and converted FP32 local model files

No NVIDIA GPU is needed for M1. CMake fetches pinned SentencePiece and nlohmann/json sources into the
build directory. Python packages are installed only in the project `.venv`.

## Reproduce M1

```bash
mise trust .mise.toml
mise exec -- uv sync --all-groups
mise exec -- uv run python scripts/prepare_model.py --include-weights
mise exec -- uv run python scripts/convert_model.py
mise exec -- uv run python scripts/export_tokenizer_golden.py
mise exec -- uv run python scripts/export_operator_golden.py
mise exec -- uv run python scripts/export_golden.py

cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure
```

The download script refuses files whose size or SHA-256 differs from
`config/model_manifest.json`. Converted weights and full-model golden traces are generated under
ignored `models/` and `data/generated/` paths. Small tokenizer and operator test corpora are tracked
so their expected reference values remain reviewable.

For a direct greedy-generation smoke run after preparation:

```bash
./build/cpu-debug/tinyllama_reference \
  models/tinyllama-chat-v1.0 \
  "The capital of France is" \
  1
```

The executable reports structured error codes for invalid format, checksum, config, shape, dtype,
OOM budget, bounds, tokenizer, and numerical failures. It does not abort for recoverable input or
model-validation errors.

The exact TLIEWGT contract and numerical reference decision are recorded in
`docs/adr/0001-m1-reference-and-weight-format.md`.

## Verify M2 on the RTX 3080 Laptop

The CUDA toolkit, driver, Compute Sanitizer, and Nsight tools are machine prerequisites and are not
installed by this project. Prepare the already-pinned FP16 artifact, then build and run the real GPU
checks:

```bash
mise exec -- uv run python scripts/convert_model_fp16.py
cmake --preset cuda-release
cmake --build --preset cuda-release
ctest --preset cuda-release --output-on-failure
compute-sanitizer --tool memcheck ./build/cuda-release/tests/kernel_tests
mise exec -- uv run python scripts/compare_logits.py
mise exec -- uv run python scripts/benchmark.py --power-mode "AC/high-performance"
mise exec -- uv run python scripts/profile_cuda.py --tool both
```

`benchmark.py` requires the exact contexts 128/512/2048/4096 and batch 1, runs full-workload
warmups, and records TTFT, TPOT, throughput, in-engine peak allocation, CUDA toolkit/runtime, and
one-second GPU temperature/clock/P-state/utilization samples while each workload is active. It
rejects incomplete or inconsistent workload and microbenchmark results, then writes JSON, CSV, and
Markdown under the ignored `benchmarks/results/` directory. The 4096-token case explicitly
extrapolates RoPE beyond the model's 2048-token training limit and is not a quality claim. Kernel
contracts, tolerances, timing boundaries, and profiling choices are recorded in
`docs/adr/0002-m2-fp16-cuda-contracts.md`.

The report also pins the prompt seed text and token IDs, deterministic context-construction rule,
greedy sampling, and the exact compute/transfer/wall-clock boundaries used by each metric.

A passing benchmark process means only that the context-128 throughput threshold passed and the
Mac-generated source snapshot stayed byte-for-byte unchanged throughout the run. The snapshot also
records Mac Git metadata, but remains reproducible for this pre-commit bootstrap repository and does
not require `.git` in WSL. Review the report's in-workload thermal/clock ranges before accepting M2;
the report marks that review as required rather than inferring it from an idle snapshot.

## Verify M3 W8A16 on the RTX 3080 Laptop

Create the deterministic per-output-channel INT8 artifact, then compare it with the same FP32 and
FP16 references and fixed workloads:

```bash
mise exec -- uv run python scripts/convert_model_int8.py
mise exec -- uv run python scripts/compare_int8.py
mise exec -- uv run python scripts/benchmark_int8.py \
  --contexts 128 4096 --output-tokens 32 --warmup 3 --samples 10 \
  --power-mode "AC/Balanced"
mise exec -- uv run python scripts/run_cuda_memcheck.py
mise exec -- uv run python scripts/profile_cuda.py --tool both --mode int8
```

The INT8 artifact quantizes transformer rank-two matrices, stores one FP32 scale per output row, and
retains the embedding, `lm_head`, RMSNorm vectors, and activations in FP16. `benchmark_int8.py`
accepts either at least 20% decode
throughput gain or at least 25% peak-device-memory reduction, but also requires a stable device
allocation count throughout every measured 128/4096 workload. A memory-only pass is reported as
such; it is not described as an INT8 speedup. Format, kernel, accuracy, and profiling contracts are
recorded in `docs/adr/0003-m3-int8-weight-only.md`.

## Verify M4 batching and serving on the RTX 3080 Laptop

The CUDA server keeps one shared FP16 weight store, four execution rows, and twenty independently
reclaimable KV slots. It listens only on loopback by default and implements `/v1/models`,
`/v1/chat/completions`, SSE streaming, cancellation, timeout, temperature/top-p sampling, and
request/scheduler metrics.

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release
ctest --preset cuda-release --output-on-failure
python3 scripts/openai_smoke_test.py
python scripts/benchmark_batch.py --power-mode "AC/high-performance" \
  --output benchmarks/results/<run>/batch.json
python3 scripts/run_cuda_memcheck.py \
  --target build/cuda-release/benchmarks/batch_benchmark -- --test
```

`openai_smoke_test.py` starts the real server and requires streaming plus non-streaming responses,
twenty concurrent complete/cancel/timeout outcomes, no hung future, and zero KV blocks after drain.
The formal batch benchmark uses context 128, 32 output tokens, three warmups, ten samples, identical
greedy tokens, and requires Batch 4 total throughput to be at least 1.5 times Batch 1. Scheduler,
KV, timing, and protocol choices are recorded in `docs/adr/0004-m4-batching-server.md`.

## Verify M5 stability and comparison

M5 adds a mixed short/medium/long-context service load, cross-engine adapters, and both memcheck and
racecheck coverage. The formal stability duration is exactly 1800 seconds with concurrency 20:

```bash
python3 scripts/load_test.py --concurrency 20 --duration 1800 \
  --power-mode "AC/high-performance" \
  --output benchmarks/results/<run>/stability.json

python3 scripts/run_cuda_memcheck.py --tool memcheck \
  --target build/cuda-release/benchmarks/batch_benchmark -- --test
python3 scripts/run_cuda_memcheck.py --tool racecheck \
  --target build/cuda-release/tests/kernel_tests
```

`load_test.py` rejects crashes, unexpected HTTP failures, NaN/Inf response metrics, more than 64
MiB of settled GPU-memory growth, or nonzero queued/active/KV counts after drain. Cancellation and
timeout are expected terminal outcomes, not silently ignored errors. The formal runner hard-rejects
any duration other than 1800 seconds or concurrency other than 20, so a shortened diagnostic cannot
be labeled as an M5 pass.

The comparison uses context 128, 32 output tokens and greedy sampling for PyTorch FP16, a pinned
llama.cpp FP16 CUDA build, TLIE FP16, and TLIE hybrid INT8. Generate the engine-specific JSON files,
then create all three report formats with one command:

Model preparation remains on the Mac. The verified llama.cpp source commit is
`571d0d540df04f25298d0e159e520d9fc62ed121`; its `convert_hf_to_gguf.py --outtype f16` command writes
the GGUF into a Mac user cache, and that model artifact is copied one way to the WSL user cache.
The WSL checkout is never used to convert or edit source. Before the external CUDA build and every
benchmark, run `scripts/sync_to_wsl.sh` as described above. Build llama.cpp in its WSL user cache
with `GGML_CUDA=ON`, `GGML_CUDA_F16=ON`, `LLAMA_CURL=OFF`, and `BUILD_SHARED_LIBS=OFF`; do not install
it globally.

```bash
# Mac model preparation (the model and GGUF remain untracked artifacts)
LLAMA_COMMIT=571d0d540df04f25298d0e159e520d9fc62ed121
MAC_LLAMA_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/tlie-baselines/llama.cpp-${LLAMA_COMMIT}"
git init "$MAC_LLAMA_ROOT"
git -C "$MAC_LLAMA_ROOT" remote add origin https://github.com/ggml-org/llama.cpp.git
git -C "$MAC_LLAMA_ROOT" fetch --depth 1 origin "$LLAMA_COMMIT"
git -C "$MAC_LLAMA_ROOT" checkout --detach FETCH_HEAD
PYTHONPATH="$MAC_LLAMA_ROOT/gguf-py" uv run python "$MAC_LLAMA_ROOT/convert_hf_to_gguf.py" \
  models/tinyllama-chat-v1.0 --outfile "$MAC_LLAMA_ROOT/tinyllama-f16.gguf" --outtype f16

# Copy only the generated model artifact to the WSL user cache.
ssh my-wsl 'mkdir -p "$HOME/.cache/tlie-baselines"'
rsync --archive --progress "$MAC_LLAMA_ROOT/tinyllama-f16.gguf" \
  my-wsl:.cache/tlie-baselines/tinyllama-f16.gguf

# WSL user-cache build; run the mandatory Mac source sync immediately before this CUDA build
LLAMA_COMMIT=571d0d540df04f25298d0e159e520d9fc62ed121
LLAMA_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/tlie-baselines/llama.cpp-${LLAMA_COMMIT}"
cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build-cuda" \
  -DGGML_CUDA=ON -DGGML_CUDA_F16=ON -DLLAMA_CURL=OFF \
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_ROOT/build-cuda" -j2 \
  --target llama-server llama-quantize

# WSL benchmark commands, after a fresh Mac source sync
.venv/bin/python scripts/benchmark_pytorch.py \
  --power-mode "AC/high-performance" \
  --output benchmarks/results/<run>/pytorch.json
python3 scripts/benchmark_llamacpp.py \
  --server "$HOME/.cache/tlie-baselines/llama.cpp-571d0d540df04f25298d0e159e520d9fc62ed121/build-cuda/bin/llama-server" \
  --model "$HOME/.cache/tlie-baselines/tinyllama-f16.gguf" \
  --llama-commit 571d0d540df04f25298d0e159e520d9fc62ed121 \
  --reference-report benchmarks/results/<run>/pytorch.json \
  --power-mode "AC/high-performance" \
  --output benchmarks/results/<run>/llama.json
```

```bash
python3 scripts/compare_engines.py \
  --tlie-report benchmarks/results/<tlie>/report.json \
  --tlie-int8-report benchmarks/results/<tlie-int8-cold>/report.json \
  --batch-report benchmarks/results/<batch>/batch.json \
  --pytorch-report benchmarks/results/<pytorch>.json \
  --llama-report benchmarks/results/<llama>.json \
  --output-dir benchmarks/results/<comparison>
```

The aggregator fails if an engine is absent, context/output metadata differs, the inputs come from
different source-tree snapshots, or the four engines produce different 32-token greedy sequences
for the fixed repeated-token prompt. The shared source-tree SHA-256 is written into JSON, CSV, and
Markdown so a report cannot silently combine stale engine runs. Peak-memory measurement sources are
preserved in the input reports; llama.cpp process-wide `nvidia-smi` memory is not presented as
allocator-local memory. Exact contracts and limitations are recorded in
`docs/adr/0005-m5-stability-and-comparison.md`.

It also requires three warmups, ten samples, the pinned prompt and model checksum, unchanged source
snapshots, an in-workload GPU monitor, and the pinned llama.cpp commit. PyTorch samples GPU state
before/after and after every warmup or measured Batch 1/4 workload; endpoint-only thermal evidence is
not accepted by the aggregator. The llama.cpp adapter verifies that the executable belongs to a
clean checkout at that commit and measures Batch 4 with the same three warmups and ten samples.

`--tlie-int8-report` lets the INT8 row come from a separately cooled, INT8-first run instead of a
row measured after FP16. Each row retains its observed software thermal-slowdown states and a
`thermal_clean` flag; a thermally active row remains correctness/memory evidence but is not accepted
as a clean performance baseline. TLIE currently has no Batch 4 INT8 path, so that cell is reported as
unavailable rather than borrowed from FP16.
