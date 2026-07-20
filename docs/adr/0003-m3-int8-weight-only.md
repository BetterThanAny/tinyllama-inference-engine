# ADR 0003: M3 W8A16 format, kernels, and acceptance boundary

- Status: Implemented; acceptance requires RTX 3080 Laptop evidence
- Date: 2026-07-19

## Context

M3 must reduce memory or improve throughput without weakening the M2 numerical, ownership, or
benchmark contracts. The designated Ampere GPU supports efficient INT8 arithmetic, but an honest
weight-only implementation must still include FP16 activation conversion and FP32 accumulation.
Its speed therefore remains a measured property rather than an architectural claim.

## Decision

The M3 representation is symmetric W8A16 with one FP32 scale per output row. Rank-two transformer
matrices are quantized independently with `scale = max(abs(row)) / 127`; zero rows use scale one.
The token embedding and `lm_head` remain FP16 because the all-matrix artifact passed a weak
first-step metric while visibly repeating `France is Paris` in the fixed generation corpus.
RMSNorm vectors also remain FP16. The mixed TLIEWGT file uses dtype code 3 for signed INT8 payloads and
stores each scale vector as an ordinary FP32 tensor named `<weight>.scale`. The manifest pins both
the original model SHA-256 and the complete converted-file SHA-256. The loader validates those
external pins, record dtype, rank, dimensions, byte counts, tensor digests, and the one-scale-per-row
relationship before allocating or uploading device storage.

The runtime dispatches rank-two W8A16 tensors to a custom row-parallel GEMV. Threads accumulate
`int8 * fp16` products in FP32, reduce within a block, apply the row scale once, and store FP16.
An element-parallel dequantization kernel remains covered for INT8 embedding-row contracts, but the
accepted artifact dispatches its protected embedding through the FP16 path. Activations, KV Cache,
RMSNorm weights, intermediate workspaces, and logits remain FP16. This is not an INT8-activation or
Tensor Core GEMM implementation, and no claim of INT8 throughput acceleration follows from its name.

The converter is deterministic: rebuilding without the explicit bootstrap option must reproduce
the pinned artifact. Python regression tests cover quantization reconstruction, invalid rank, report
acceptance, numerical metrics, and CUDA subprocess wrapping. CUDA tests compare both new kernels to
CPU references and cover invalid embedding bounds. The end-to-end accuracy report compares FP16 and
INT8 logits against the M1 FP32 reference, reports maximum/mean/RMS error, cosine similarity, top-1,
Jensen-Shannon divergence, a top-1 perplexity proxy, greedy common prefix, and token agreement. The
accepted gate requires both FP16 and INT8 to match all fixed greedy tokens exactly; cosine or a short
prefix alone cannot qualify a visibly degraded sequence.

The paired performance run fixes contexts 128 and 4096, batch one, greedy sampling, 32 output
tokens, three full-workload warmups, and ten samples. It runs FP16 and INT8 in the same report,
records one-second thermal/clock samples, validates the model hashes and unchanged Mac source
snapshot, and requires either at least 20% context-128 decode-throughput gain or at least 25%
context-128 peak-device-memory reduction. Allocation counters are sampled immediately before and
after every measured workload and must remain equal. KV bytes are reported independently for both
contexts and both dtypes.

Nsight Systems profiles the real INT8 end-to-end command and its NVTX prefill/decode/sampling
ranges. Nsight Compute profiles one launch of each microbenchmark family, including the W8A16 GEMV
and embedding-row kernels. Compute Sanitizer memcheck covers the CUDA regression executable. These
artifacts determine whether a speed result is a finding or an explicitly retained non-finding.

## Consequences

- The hybrid INT8 artifact is ignored by Git and is materially smaller than FP16, while FP16 KV and
  workspace memory are unchanged. Protecting the two vocabulary-facing matrices trades some of the
  all-matrix memory saving for an exact fixed-corpus generation result.
- Per-output-channel scales improve accuracy and make the dequantization contract inspectable, but
  add one scale load per output row.
- The scalar/block-reduction GEMV prioritizes correctness and memory reduction. If profiling shows
  that conversion, reduction, or launch overhead prevents a speedup, the implementation remains a
  valid memory optimization and the throughput result is reported as a non-finding.
- KV Cache is still allocated once at model creation for the declared maximum sequence length. M3
  does not change its numerical representation or ownership contract; stable allocation counters
  and sanitizer evidence are required instead of inferring safety from successful generation.
- Fusion is accepted only when profiler evidence identifies a material boundary and regression
  tests preserve the unfused reference. Otherwise the absence of a justified fusion is documented
  as a non-finding rather than adding unmeasured complexity.
