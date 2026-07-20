# ADR 0001: M1 CPU reference and TLIEWGT format

- Status: Accepted for M1
- Date: 2026-07-13

## Context

M1 needs an implementation that can validate the pinned TinyLlama model without CUDA, expose
readable reference operators, and reject damaged or incompatible weights before execution. Loading
the upstream BF16 safetensors directly from C++ would couple the runtime to a Python-oriented format
and would not establish the FP32 reference representation used for numerical comparison.

## Decision

The source model is `TinyLlama/TinyLlama-1.1B-Chat-v1.0` at immutable revision
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`. `config/model_manifest.json` pins every required
file's byte size and SHA-256. The original `model.safetensors` remains an uncommitted local asset.

`scripts/convert_model.py` converts every upstream tensor to contiguous little-endian FP32 and
writes TLIEWGT version 1. The C++ M1 runtime memory-maps this file and uses read-only tensor views.
The conversion deliberately trades disk and memory bandwidth for a transparent reference dtype;
it is not the future FP16 CUDA storage format.

The byte stream is:

1. File header: eight-byte `TLIEWGT\0` magic, little-endian `uint32` version, little-endian
   `uint32` tensor count, and the 32 raw bytes of the source safetensors SHA-256.
2. One record per tensor: little-endian `uint16` UTF-8 name length, `uint8` dtype (`1` is FP32),
   `uint8` rank, `uint64` payload bytes, 32-byte payload SHA-256, `rank` little-endian `uint64`
   dimensions, and the UTF-8 name.
3. Zero padding to the next 64-byte file offset, followed by the contiguous row-major payload.
4. No duplicate names and no trailing bytes are permitted.

The loader validates an externally pinned full-file SHA-256 before trusting the stream, then checks
magic, version, bounds, arithmetic overflow, dtype, dimensions, byte count, source checksum, each
tensor checksum, the exact 201-tensor set, and every expected model shape. The external digest is
the trust root; embedded tensor digests alone are not accepted as proof of file identity. Its
configurable pre-map byte budget provides a deterministic structured OOM failure path. Actual
allocation failures are also translated to `out_of_memory` errors.

The C++ reference executes one token at a time with explicit contiguous positions. KV storage is
owned by the model, preallocated for a declared maximum sequence length, bounds checked, and reset
explicitly. The operators are FP32 RMSNorm, half-split Llama RoPE, stable Softmax, grouped-query
causal attention, SiLU-gated MLP, and deterministic argmax with lowest-ID tie breaking. Apple
Accelerate supplies only the matrix-vector primitive on macOS; the layer structure and all other
operators remain visible C++ reference code.

Python/Transformers with eager FP32 attention is the external golden-data producer. M1 compares the
last prompt position's embedding input and every layer's normalization, attention output,
post-attention normalization, MLP output, and layer output, followed by final normalization and all
logits, using recorded absolute and relative tolerances. Greedy decoding must then match all 32
token IDs exactly.

## Consequences

- The converted file is approximately twice the BF16 source size and is excluded from Git.
- Full-model CPU verification is correctness-oriented and intentionally not a performance claim.
- CUDA, FP16 storage, kernel layouts, quantization, and service protocols remain undecided and out
  of M1 scope.
- SentencePiece and nlohmann/json are pinned CMake source dependencies. PyTorch, Transformers,
  safetensors, NumPy, SentencePiece, and protobuf are pinned project-local uv dependencies.
