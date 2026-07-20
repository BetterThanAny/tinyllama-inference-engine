# TinyLlama CUDA 推理与服务引擎实施计划

## 1. 项目定位

构建一个可验证、可量化、可服务化的小型 LLM 推理引擎。核心模型为 TinyLlama-1.1B，覆盖权重加载、tokenizer、KV Cache、FP16 CUDA 算子、INT8 weight-only、批处理调度、流式生成和 OpenAI-compatible API。

项目重点从“手写算子”扩展为“正确性、延迟、吞吐、显存和部署成本均有证据的推理服务”。

### 目标岗位信号

- C++/CUDA 推理路径和性能分析
- KV Cache、prefill/decode、量化与批处理
- 数值对齐和可重复 benchmark
- OpenAI-compatible serving
- GPU 资源管理、取消、超时和观测
- 理解 LLM 应用后端依赖的推理成本

### 硬件边界

- 通用代码、文档、CPU reference 和静态测试可在 Mac 开发。
- 所有 CUDA 正确性、性能、显存和稳定性结论必须在 16GB RTX 3080 Laptop 上运行。
- Mac 本地 checkout 是源码唯一真源；CUDA 验证前通过 `scripts/sync_to_wsl.sh` 将源码镜像到
  WSL 的 `~/tinyllama-inference-engine`，再通过 SSH alias `my-wsl` 执行远端命令。远端源码
  不直接编辑，`.git`、构建目录、模型、生成数据和 benchmark 产物不参与同步。
- 3080 Laptop 属于 Ampere 路线，核心目标为 FP16 和 INT8，不把缺少原生优势的 FP8 作为验收项。
- Benchmark 必须记录 GPU 型号、驱动、CUDA、功耗模式、温度、时钟、batch 和上下文长度。

### 非目标

- 不与 vLLM 竞争通用模型覆盖或集群规模。
- 不实现训练、LoRA 训练或分布式推理。
- 不为追求高数字跳过数值校验。
- 不在 Mac 或模拟环境中声称 CUDA 性能通过。

## 2. GitHub 调研基线

- [llama.cpp](https://github.com/ggml-org/llama.cpp)：参考量化、CUDA kernel、CPU/GPU 混合路径和 OpenAI-compatible server。
- [vLLM](https://github.com/vllm-project/vllm)：参考 PagedAttention、continuous batching、prefix cache 和服务指标。
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)：作为 NVIDIA GPU 优化与 benchmark 对照。
- [llama2.c](https://github.com/karpathy/llama2.c)：参考最小、透明的模型执行路径。

## 3. 推荐技术栈与目录

- C++20、CUDA、CMake
- cuBLAS，必要时使用 CUTLASS 做对照而不是隐藏全部实现
- Python 仅用于模型转换、golden 数据和 benchmark 汇总
- FastAPI 不作为核心 server；优先 C++ HTTP server 或薄 Python binding，选定后记录 ADR
- GoogleTest/Catch2 二选一
- Nsight Systems、Nsight Compute
- Docker 作为可选 CUDA 运行环境

```text
include/
src/
  model/
  tokenizer/
  kernels/
  runtime/
  scheduler/
  server/
tests/
benchmarks/
scripts/
docs/
```

## 4. 里程碑

### M1：正确性基线与可重复构建

#### 工作内容

- 固定 TinyLlama 权重、配置和 tokenizer 版本。
- 明确权重文件格式、校验和、shape 和 dtype。
- 建立 CPU reference path 或可读的逐层 reference harness。
- 从 PyTorch 导出输入、各层关键张量、logits 和 greedy token golden data。
- 测试 RMSNorm、RoPE、Softmax、Attention、MLP、采样器和 KV Cache。
- 增加损坏权重、非法 shape、OOM 和 tokenizer 边界错误。

#### 退出条件

- Tokenizer 固定语料的 token ID 与参考实现完全一致。
- FP32 reference 的关键张量和 logits 在约定容差内通过。
- Greedy 模式前 32 个 token 与参考实现一致。
- CPU tests 可在无 NVIDIA GPU 的环境运行。

#### 实施状态（2026-07-13）

**已完成。** 本段状态只覆盖 M1；M2 的独立状态记录在下一节，M3 及后续里程碑尚未开始。

- 模型固定为 `TinyLlama/TinyLlama-1.1B-Chat-v1.0` revision
  `fe8a4ea1ffedaf415f4da2f062534de366a451e6`。原始 safetensors 大小为
  `2,200,119,864` 字节，SHA-256 为
  `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`。
- TLIEWGT v1 将 201 个权重转换为 little-endian FP32，并对来源、每个 tensor 和完整转换文件做
  SHA-256 校验。转换文件大小为 `4,400,219,264` 字节，SHA-256 为
  `277f8aa3757b47b208c02682e851590bc42e819279251ac186406bd67b05beaf`。格式与 loader
  contract 记录在 `docs/adr/0001-m1-reference-and-weight-format.md`。
- 固定 tokenizer corpus 共 7 个正常/空输入/Unicode/空白边界 case，与 Transformers slow
  Llama tokenizer (`legacy=False`) 的 token ID 完全一致；另外验证非法 UTF-8、输入长度上限和
  非法 token ID 失败路径。
- PyTorch eager FP32 reference 导出 prompt input、22 层关键张量、final norm 和完整 logits，
  共 113 个 tensor。C++ reference 在 `atol=1e-4, rtol=1e-4` 下全部通过，观测到的最大绝对
  误差为 `6.60419e-05`；greedy 前 32 个 token 全部逐项一致且每步通过 NaN/Inf 检查。
- loader/config/KV/sampling 测试覆盖正常路径，以及损坏 checksum、截断格式、非法 shape、
  固定架构不匹配、确定性 OOM budget、KV 越界、NaN/Inf 和 tokenizer 边界失败路径。
- `cpu-debug` 使用 Apple Clang 17 的 ASan/UBSan，在无 NVIDIA GPU 的 macOS arm64 环境实际
  运行 4 个测试目标；4/4 通过，无 disabled、skip、`Not Run` 或 0-tests 情况。

实际验证命令与结果：

```bash
mise exec -- uv run python scripts/prepare_model.py --include-weights  # 全部 5 个文件校验通过
mise exec -- uv run python scripts/convert_model.py                    # 固定 size/SHA-256 通过
mise exec -- uv run python scripts/export_tokenizer_golden.py         # 7 cases
mise exec -- uv run python scripts/export_operator_golden.py          # PyTorch FP32 goldens
mise exec -- uv run python scripts/export_golden.py --threads 8       # 113 tensors, 32 tokens

mise exec -- uv run ruff check scripts                                # 通过
mise exec -- uv run mypy scripts                                      # 7 files，无问题
xcrun clang-format --dry-run --Werror app/*.cpp include/tlie/*.hpp src/*.cpp tests/*.cpp tests/*.hpp
                                                                        # 通过
cmake --preset cpu-debug                                                # 配置成功
cmake --build --preset cpu-debug                                       # 构建成功
ctest --preset cpu-debug --output-on-failure                            # 4/4 通过，18.30 秒
./build/cpu-debug/tinyllama_reference models/tinyllama-chat-v1.0 \
  "The capital of France is" 1 3681                                   # token 3681，解码为 Paris
```

#### M1 复审与缺口修复（2026-07-15）

复审先以只读方式重新检查全部源码、测试、CTest 清单、已有 golden/验证记录和 Git 状态。仓库
尚无 commit，全部 62 个项目文件均为 untracked，因此普通 `git diff` 为空，不能作为“没有改动”
或历史基线证据。

复审确认一个退出条件相关缺口：原 loader 只校验文件内嵌的来源/tensor digest，CLI 没有传入
manifest 固定的完整 TLIEWGT SHA-256。使用 APFS clone 修改 tensor、同步重算内嵌 digest 后，旧
CLI 仍会执行该自洽篡改文件。现已把外部完整文件 SHA-256 设为 CLI 和集成测试的执行前信任根，
并增加可重复的单元回归；同一篡改文件现在以结构化 `checksum_mismatch` 拒绝。

退出条件复审结论：

| M1 退出条件 | 状态 | 证据 |
|---|---|---|
| 固定 tokenizer 语料 token ID 完全一致 | verified | tokenizer 28 checks；7-case golden 原样重建 |
| FP32 关键 tensor/logits 在容差内 | verified | 113 tensors；483 checks；最大绝对误差 `6.60419e-05` |
| Greedy 前 32 token 一致 | verified | 集成测试逐 token 比较及 NaN/Inf 检查 |
| 无 NVIDIA GPU 的 CPU tests 可运行 | verified | macOS arm64，ASan 启用，CTest 4/4 |
| skipped/0-tests/环境 fallback | non-finding | CTest JSON 恰有 4 个 enabled tests；无 skip/Not Run/0 tests |

修复后的实际复验：

```bash
mise exec -- uv lock --check                                      # 48 packages，锁文件有效
mise exec -- uv run ruff check scripts                            # 通过
mise exec -- uv run mypy scripts                                  # 11 files，无问题
xcrun clang-format --dry-run --Werror app/*.cpp app/*.cu include/tlie/*.hpp \
  include/tlie/cuda/*.hpp src/*.cpp src/cuda/* tests/*.cpp tests/*.hpp \
  tests/cuda/*.cu benchmarks/*.cu                                  # 通过
cmake --preset cpu-debug                                           # 配置成功，ASan/UBSan 开启
cmake --build --preset cpu-debug                                  # 构建成功
ctest --preset cpu-debug --output-on-failure                       # 4/4，38.84 秒
ctest --test-dir build/cpu-debug --show-only=json-v1              # 4 enabled，0 disabled
./build/cpu-debug/tlie_m1_integration_tests                        # 483 checks，113 traces
./build/cpu-debug/tinyllama_reference models/tinyllama-chat-v1.0 \
  "The capital of France is" 1 3681                               # token 3681，Paris
```

#### M1 本轮再审与第二个缺口修复（2026-07-16）

本轮重新完整读取当前指令、计划、64 个项目文件、测试、验证记录和 Git 状态，并从固定模型文件
校验开始重跑 M1。仓库仍无 commit，`git ls-files` 为 0，全部项目内容均为 untracked；因此
`git diff` 和 `git diff --cached` 为空只表示 Git 没有可比较的已跟踪基线，不能证明工作树未改动。

只读审计发现第二个错误路径缺口：`tinyllama_reference` 用未捕获的 `std::stoi` 解析
`MAX_TOKENS` 和可选 expected token，非法输入会触发 `std::invalid_argument` 并 abort，违反
“可恢复错误返回结构化错误”的不变量；expected token 还会在约 28 秒模型加载后才失败。新增的
两个 CLI regression tests 在修复前为 0/2，均观察到 subprocess aborted。现改为严格
`std::from_chars`，在模型加载前解析全部整数，并稳定返回 JSON `invalid_argument`；修复后目标
测试 2/2、全套 CTest 6/6 通过。

退出条件再审结论：

| M1 退出条件/审计项 | 状态 | 本轮证据 |
|---|---|---|
| 固定 tokenizer 语料 token ID 完全一致 | verified | tokenizer 28 checks；固定 7-case corpus 校验通过 |
| FP32 关键 tensor/logits 在容差内 | verified | 113 traces；483 checks；最大绝对误差 `6.60419e-05` |
| Greedy 前 32 token 一致 | verified | 集成测试逐 token 比较；smoke 首 token `3681` / `Paris` |
| 无 NVIDIA GPU 的 CPU tests 可运行 | verified | macOS arm64 Debug，ASan/UBSan 开启，CTest 6/6 |
| 可恢复 CLI 非法整数返回结构化错误 | verified | regression 先红 0/2，修复后 2/2，错误码 `invalid_argument` |
| skipped/disabled/0-tests/环境 fallback | non-finding | CTest JSON 恰有 6 个 enabled tests；无 disabled、skip、Not Run、0 tests 或 CUDA fallback |

本轮实际命令与结果：

```bash
mise exec -- uv run python scripts/prepare_model.py --include-weights  # 5 个固定资产 size/SHA-256 通过
mise exec -- uv run python scripts/export_tokenizer_golden.py         # 重建 7 cases
mise exec -- uv run python scripts/export_operator_golden.py          # 重建 FP32 operator goldens
mise exec -- uv run python scripts/export_golden.py --threads 8       # 重建 113 tensors / 32 tokens
mise exec -- uv lock --check                                         # 48 packages，锁文件有效
mise exec -- uv run ruff format --check scripts                      # 通过
mise exec -- uv run ruff check scripts                               # 通过
mise exec -- uv run mypy scripts                                     # 11 files，无问题
xcrun clang-format --dry-run --Werror app/*.cpp app/*.cu include/tlie/*.hpp \
  include/tlie/cuda/*.hpp src/*.cpp src/cuda/* tests/*.cpp tests/*.hpp \
  tests/cuda/*.cu benchmarks/*.cu                                    # 通过
cmake --preset cpu-debug && cmake --build --preset cpu-debug         # 配置/构建通过
ctest --preset cpu-debug --output-on-failure --verbose               # 6/6，32.41 秒
ctest --test-dir build/cpu-debug --show-only=json-v1                 # 6 enabled，0 disabled
otool -L build/cpu-debug/tlie_m1_integration_tests                   # 链接 ASan runtime
./build/cpu-debug/tinyllama_reference models/tinyllama-chat-v1.0 \
  "The capital of France is" 1 3681                                 # token 3681，Paris
```

结论：M1 本轮发现的缺口已修复，四项必需退出条件全部 `verified`；M1 没有剩余必需
`failed` 或 `unverified` 项，因而允许继续且仅继续 M2。

### M2：FP16 CUDA 路径与 Profiling

#### 工作内容

- 实现/整理 FP16 RMSNorm、RoPE、Softmax、GEMV/GEMM、Attention、KV Cache 更新。
- 每个 kernel 提供 correctness test 和 microbenchmark。
- 使用 CUDA event 做 kernel 计时，避免把初始化和 IO 混入。
- 用 Nsight Systems 分离 prefill、decode、sampling 和同步时间。
- 用 Nsight Compute 查看带宽、occupancy 和 launch bottleneck。
- 建立 batch=1、context=128/512/2048/4096 的基线。

#### 退出条件

- 所有 FP16 kernel 与 PyTorch/cuBLAS reference 在约定容差内通过。
- 上下文 128、512、2048、4096 无 NaN、越界和崩溃。
- Benchmark 报告可一条命令重建。
- 固定测试条件下 FP16 decode 不低于已有约 90 tok/s 基线；若硬件/环境不同必须重新定义基线。

#### 实施状态（2026-07-15）

**未完成；等待指定 RTX 3080 Laptop 的真实 CUDA 验收。不得把以下本地结果解释为 M2
通过。**

已实现但尚待 GPU 编译/运行验证的范围：

- 固定 FP16 TLIEWGT 转换和完整文件/来源/tensor SHA-256 验证；201 tensors，文件大小
  `2,200,122,496` 字节，SHA-256
  `1038f532a16e316f1953cf721cb2a54783cd1327b6a22e45ce71dcc2c574ab63`。
- FP16 RMSNorm、half-split RoPE、stable Softmax、cuBLAS GEMV/GEMM、grouped-query decode
  Attention、KV update、SiLU 和 residual add；所有权、shape/bounds 和错误传播 contract 记录于
  `docs/adr/0002-m2-fp16-cuda-contracts.md`。
- 预分配权重/workspace/KV 的 token-by-token CUDA model，无 per-token device allocation；CUDA
  event 分离 compute/transfer，host monotonic clock 记录 wall/sampling，NVTX 标记 warmup、prefill、
  decode 和 sampling。
- 真实 CUDA kernel reference tests（包括 KV 越界、NaN 和 4096 Attention 边界）、每个 M2
  kernel family 的 CUDA-event median/p95 微基准、32,000 logits + 32 greedy token 对齐脚本、
  Nsight Systems/Compute orchestration，以及一条命令生成 JSON/CSV/Markdown 的 batch-1 四上下文
  报告。
- 4096 是显式 RoPE extrapolation 的内存安全/性能压力项，不是生成质量声明；KV 容量额外允许
  最多 256 个 decode token。90 tok/s 门槛使用 context=128 的端到端 median decode（包含 GPU、
  transfer、同步、host conversion 和 sampling），同时单独报告 compute-only 数值。

Mac 上已实际通过的非 CUDA 验证：

```bash
mise exec -- uv run python scripts/convert_model_fp16.py            # 固定 size/SHA-256 通过
mise exec -- uv lock --check                                       # 48 packages
mise exec -- uv run ruff check scripts                             # 通过
mise exec -- uv run mypy scripts                                   # 11 files，无问题
xcrun clang-format --dry-run --Werror app/*.cpp app/*.cu include/tlie/*.hpp \
  include/tlie/cuda/*.hpp src/*.cpp src/cuda/* tests/*.cpp tests/*.hpp \
  tests/cuda/*.cu benchmarks/*.cu                                   # 通过
jq empty CMakePresets.json config/model_manifest.json              # 通过
cmake --preset cpu-debug && cmake --build --preset cpu-debug       # 通过
ctest --preset cpu-debug --output-on-failure                        # 4/4，38.84 秒
```

GPU 必需项仍为 `unverified`：`cuda-release` 编译、CUDA CTest、Compute Sanitizer、全模型 logits/
greedy smoke、128/512/2048/4096 end-to-end、Nsight Systems/Compute、温度/时钟/功耗/显存元数据和
90 tok/s 门槛。当前 Mac 没有 `nvcc`、`nvidia-smi`、`compute-sanitizer`、`nsys` 或 `ncu`；本地
CUDA configure 如预期报 `Failed to find nvcc`。`ssh -o BatchMode=yes -o ConnectTimeout=5 xs-wsl`
连接指定主机 `192.168.1.123:22` 超时，因此本次没有任何 CUDA 假通过、skip 或 fallback 结果。

#### M2 本轮实现与验证边界（2026-07-16）

在不进入 M3 的前提下，本轮补齐 M2 benchmark/profiling 的验收契约：每次 warmup 现在执行与
测量样本完全相同的完整 context + 32 output-token workload；结果新增 TTFT、TPOT、首次 sampling、
模型常驻分配和引擎峰值 device allocation；报告记录 `nvcc` toolkit version，并在每个微基准/
context 子进程存活期间每秒采集显存、温度、SM/显存时钟、P-state 和利用率范围。报告生成器会
拒绝错误 GPU/SM、sample 数、workload 元数据、模型 hash、显存关系、kernel family/context
matrix、NaN/Inf 或运行期元数据漂移。
Markdown 尾部指标改为 p95 decode wall latency，不再把有利方向的 p95 tok/s 当作 tail。Nsight
Compute 仍使用 full metric set，但只 profile 每个 kernel family 的一个 launch，避免把 50 个计时
样本全部 profile。脚本自动验收同时要求 90 tok/s threshold 和运行前后同一个 clean、non-null Git
commit；报告仍明确要求人工审阅负载期温度/时钟范围，不能用自动绿灯单独接受 M2。
四个 context 还会固定并交叉校验 prompt seed 文本、实际 token IDs、确定性重复规则、greedy
sampling 和 compute/transfer/wall-clock 计时边界，避免不同 workload 被误归为同一基线。

本机实际通过的 M2 非 CUDA 验证：

```bash
mise exec -- uv run ruff format --check scripts tests/test_benchmark_report.py  # 通过
mise exec -- uv run ruff check scripts tests/test_benchmark_report.py          # 通过
mise exec -- uv run mypy scripts                                               # 11 files，无问题
mise exec -- uv run python -m unittest discover -s tests -p 'test_*.py' -v     # 9/9
xcrun clang-format --dry-run --Werror app/*.cu benchmarks/*.cu                 # 通过
jq empty CMakePresets.json config/model_manifest.json data/golden/*.json       # 通过
cmake --preset cpu-debug && cmake --build --preset cpu-debug                   # 通过
ctest --preset cpu-debug --output-on-failure --verbose                         # 6/6，32.41 秒
```

新增的 9 个 Python regression tests 覆盖报告正常路径、4096 RoPE extrapolation 边界、sample
数量不一致、非有限测量、缺失 kernel family/context matrix、Markdown tail 指标，以及负载期间
温度/时钟/P-state 范围汇总、clean Git baseline gating 和 prompt metadata 漂移。它们只验证报告
契约，不替代任何 CUDA 数值或性能实跑。

M2 退出条件当前结论：

| M2 退出条件/审计项 | 状态 | 本轮证据或缺口 |
|---|---|---|
| 所有 FP16 kernel 与 reference 在容差内 | unverified | 指定 GPU 不可达；CUDA target 尚未真实编译或运行 |
| context 128/512/2048/4096 无 NaN、越界和崩溃 | unverified | 必须在 3080 Laptop 跑 CUDA CTest、memcheck 和全模型 benchmark |
| Benchmark 报告可一条命令重建 | unverified | 生成/拒绝契约的 9 个本地测试通过，但真实 GPU 一命令报告尚未产生 |
| context=128 median FP16 decode >= 90 tok/s | unverified | 没有指定 GPU 的性能样本，不存在可接受的替代基线 |
| CUDA skipped/disabled/0-tests | unverified | `cuda-release` 未配置完成，尚不能生成 CUDA CTest inventory；preset 的 `noTestsAction=error` 防止 0-tests 绿灯 |
| CUDA backend fallback | non-finding | CMake/source 中没有 CPU/其他 backend 冒充 CUDA 的 fallback；本机直接因缺少 `nvcc` 失败 |
| GPU 环境 gating | verified | Mac `cmake --preset cuda-release` 明确报 `Failed to find nvcc`；`xs-wsl` SSH 到 `192.168.1.123:22` 再次超时 |
| 正式 benchmark 的 Git commit 基线 | failed | 仓库为 `No commits yet`、0 tracked files；报告只能记录 `commit: null` 和 dirty tree，本轮明确禁止 commit，不能把该状态接受为正式性能基线 |

2026-07-16 10:12 CST 的第三次连续环境复核仍得到相同结果：本机 `nvcc`、`nvidia-smi`、
`compute-sanitizer`、`nsys`、`ncu` 全部缺失，且 `xs-wsl` 在 8 秒连接超时后仍不可达；Git 仍为
`No commits yet`、0 tracked files。没有出现可运行 CUDA 验收的新证据。

因此 M2 仍为**未完成**，不得更新为通过；需要先由用户建立可复现的 commit 基线，并在指定 RTX
3080 Laptop 可连接后实际运行：

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release
ctest --preset cuda-release --output-on-failure
compute-sanitizer --tool memcheck ./build/cuda-release/tests/kernel_tests
mise exec -- uv run python scripts/compare_logits.py
mise exec -- uv run python scripts/benchmark.py --power-mode "AC/high-performance"
mise exec -- uv run python scripts/profile_cuda.py --tool both
```

本轮到此停止；M3 及后续代码未开始。

#### M1 恢复审计与 M2 checksum 合同修复（2026-07-16 10:43 CST）

恢复执行后先重新完整读取当前指令、计划、源码、测试和既有验证记录，再从固定资产校验开始重跑
M1。仓库仍为 `No commits yet`、0 tracked files；普通 `git diff`/`git diff --cached` 为空仍不构成
历史基线。当前 CTest JSON 恰有 6 个 enabled tests、0 disabled tests；本轮 6/6 实际运行通过，合计
604 个显式 C++ checks，未发现 skip、`Not Run`、0-tests 或环境 fallback。陈旧的
`LastTestsFailed.log` 修改时间仍停留在 09:40，早于本轮 10:36 的 `LastTest.log`，因此不把它误判为
当前失败。随后从锁定依赖重建的 `reference_trace.tliewgt` SHA-256 仍为
`81705010bccb0c310db7ed6ce57b5aa66f34bdbc2de89c48ed9d94b17282ce5a`，重建后集成测试仍为
483 checks、exit 0。

M1 退出条件恢复审计：

| M1 退出条件/审计项 | 状态 | 本轮证据 |
|---|---|---|
| 固定 tokenizer 语料 token ID 完全一致 | verified | 7-case corpus；tokenizer 28 checks |
| FP32 关键 tensor/logits 在容差内 | verified | 113 traces；483 checks；最大绝对误差 `6.60419e-05` |
| Greedy 前 32 token 一致 | verified | 集成测试逐 token 比较；CLI smoke 首 token `3681` / `Paris` |
| 无 NVIDIA GPU 的 CPU tests 可运行 | verified | macOS arm64 Debug；CTest 6/6；测试二进制链接 ASan runtime |
| CLI 非法整数结构化失败路径 | verified | 两个 failure CTest 均运行并通过，未 abort |
| skipped/disabled/0-tests/环境 fallback | non-finding | 6 enabled、0 disabled；verbose log 无 skip/`Not Run`/0-tests/fallback |

M1 本轮实际复验：

```bash
mise exec -- uv run python scripts/prepare_model.py --include-weights  # 5 个固定资产 size/SHA-256 通过
mise exec -- uv lock --check                                         # 48 packages，锁文件有效
mise exec -- uv run ruff format --check scripts tests                # 通过
mise exec -- uv run ruff check scripts tests                         # 通过
mise exec -- uv run mypy scripts                                     # 11 files，无问题
xcrun clang-format --dry-run --Werror $(fd --type f --extension cpp \
  --extension cu --extension hpp . src include app tests benchmarks) # 33 files，通过
jq empty config/model_manifest.json data/golden/*.json \
  data/generated/tinyllama-chat-v1.0/reference.json                   # 通过
cmake --preset cpu-debug && cmake --build --preset cpu-debug         # 配置/构建通过
ctest --preset cpu-debug --show-only=json-v1                          # 6 enabled，0 disabled
ctest --preset cpu-debug --output-on-failure --verbose               # 6/6；约 37.54 秒
otool -L build/cpu-debug/tlie_unit_tests \
  build/cpu-debug/tlie_m1_integration_tests                          # 两者均链接 ASan runtime
./build/cpu-debug/tinyllama_reference models/tinyllama-chat-v1.0 \
  "The capital of France is" 1                                      # token 3681；Paris
./build/cpu-debug/tlie_m1_integration_tests                            # 重建 Golden 后 483 checks，exit 0
```

M1 没有新的必需缺口，因而继续且仅继续 M2。本轮随后发现 M2 报告只检查 CUDA 子进程回报的模型
checksum 长度，没有与 `config/model_manifest.json` 固定的 source/FP16 checksum 做值相等校验。
新增 checksum-drift regression 修复前 0/1（预期的 `ValueError` 未出现）；修复后 benchmark 从
manifest 读取两个 pin，拒绝 schema、缺失/非法 checksum 和运行结果漂移，并把 manifest 路径写入报告。
新增 manifest 正常/非法路径测试后，报告合同测试为 11/11。

M2 本轮非 CUDA 验证：

```bash
mise exec -- uv run python -m unittest \
  tests.test_benchmark_report.BenchmarkReportTests.test_rejects_model_checksum_drift -v
                                                                    # 修复前 0/1，修复后 1/1
mise exec -- uv run ruff format --check scripts tests/test_benchmark_report.py  # 通过
mise exec -- uv run ruff check scripts tests/test_benchmark_report.py          # 通过
mise exec -- uv run mypy scripts                                               # 11 files，无问题
mise exec -- uv run python -m unittest discover -s tests -p 'test_*.py' -v     # 11/11，无 skip
mise exec -- uv run python scripts/convert_model_fp16.py
                                                                    # 固定 size/SHA-256 通过
cmake --preset cuda-release                                         # 预期硬失败：Failed to find nvcc
ssh -o BatchMode=yes -o ConnectTimeout=8 xs-wsl ...                 # Host is down
```

M2 退出条件本轮结论：

| M2 退出条件/审计项 | 状态 | 本轮证据或缺口 |
|---|---|---|
| 所有 FP16 kernel 与 reference 在容差内 | unverified | 指定 GPU 不可达；CUDA target 仍未真实编译/运行 |
| context 128/512/2048/4096 无 NaN、越界和崩溃 | unverified | 未运行 CUDA CTest、memcheck 或全模型四上下文 workload |
| Benchmark 报告可一条命令重建 | unverified | manifest/checksum 报告合同 11/11 通过；真实 GPU 报告仍未产生 |
| context=128 median FP16 decode >= 90 tok/s | unverified | 没有指定 GPU 性能样本 |
| CUDA skipped/disabled/0-tests | unverified | 本机 configure 在生成 CUDA CTest inventory 前因缺少 `nvcc` 硬失败 |
| CUDA backend fallback | non-finding | 本机没有 skip 或 CPU 冒充；CUDA preset 明确以非零状态失败 |
| 指定 GPU 环境 gating | verified | Mac 无 CUDA 工具；目标 ARP incomplete、ICMP 丢包，TCP/22 最终返回 `Host is down` |
| 正式 benchmark 的 clean Git commit 基线 | failed | `No commits yet`、0 tracked files；当前任务不授权 commit |

因此 M2 仍为**未完成**，没有开始 M3。待指定 RTX 3080 Laptop 在线且建立获准的可复现 Git
baseline 后，仍须实际运行上一节列出的 CUDA build、CTest、Compute Sanitizer、logits/greedy、四上下文
benchmark 和 Nsight 命令；所有必需项通过并完成人工温度/时钟审阅前不得标记完成。

#### M1 再审计与首次 RTX 3080 Laptop 实跑（2026-07-16 13:03 CST）

本轮再次从当前磁盘状态审计 M1，随后只推进 M2。M1 的固定资产、锁文件、格式、类型、Golden、
CPU build、CTest inventory 和 smoke 均重新运行：CTest 为 6 enabled、0 disabled、6/6 通过，合计
604 个 C++ checks；11/11 Python tests 通过且无 skip。日志未出现 `Not Run`、0-tests、fallback 或
环境 gating；两个测试二进制实际链接 ASan，CLI smoke 仍生成 token `3681` / `Paris`。因此 M1
四项必需退出条件继续为 `verified`，没有先修复的缺口。

指定主机 `192.168.1.222:2222` 本轮可达并确认为 16GB `NVIDIA GeForce RTX 3080 Laptop GPU`
（SM 8.6），WSL2 kernel `6.18.33.1`、driver `610.47`、CUDA toolkit `12.0.140`、runtime
`12000`。Windows 实测 `PowerOnline=true`，电源计划为 Balanced；没有改动系统电源计划、driver、
toolkit 或全局软件。WSL 必须把 `/usr/lib/wsl/lib` 放在 `LD_LIBRARY_PATH` 前端，否则动态加载到
冲突的发行版 `libcuda` 后 `cudaGetDevice` 会失败；显式使用 WSL driver library 后同一测试完整
执行。

真实 CUDA build 暴露并修复了两个此前只在 Mac 上不可见的编译缺口：显式包含定义
`CUDART_INF_F` 的 `math_constants.h`；仅对使用 nlohmann/json 的两个 `.cu` 可执行目标设置
`JSON_HAS_RANGES=0`，绕过 CUDA 12.0 NVCC 不支持其可选 ranges specialization 的问题，仍保持
C++20。另补上 residual-add kernel 的 CUDA-event microbenchmark，并让报告合同拒绝该 family
缺失。修复后的 microbenchmark 覆盖 12 rows（四个 Attention context）并实际运行 10 warmups /
50 samples；residual-add median/p95 为 `0.013312/0.015360 ms`。

本轮指定 GPU 实际验证：

```bash
cmake --preset cuda-release && cmake --build --preset cuda-release
                                                                    # 配置/构建通过
ctest --preset cuda-release --show-only=json-v1                     # 7 enabled，0 disabled
LD_LIBRARY_PATH=/usr/lib/wsl/lib ctest --test-dir build/cuda-release \
  -R '^tlie_cuda_kernel_tests$' --output-on-failure --verbose       # 1/1；233 checks
UV_PYTHON_DOWNLOADS=never mise exec -- uv run --offline \
  --with .cache/wheels/numpy-2.2.6-*.whl --no-project \
  python scripts/compare_logits.py                                  # 32 tokens / 32,000 logits 通过
LD_LIBRARY_PATH=/usr/lib/wsl/lib mise exec -- python scripts/benchmark.py \
  --power-mode "AC/Balanced (PowerOnline=true, Charging=false)"    # 报告生成；exit 2（仅 Git gate）
compute-sanitizer --tool memcheck --injection-path \
  /usr/lib/nvidia-cuda-toolkit/compute-sanitizer \
  ./build/cuda-release/tests/kernel_tests                           # 失败：首个 instrumented API 前终止
mise exec -- python scripts/profile_cuda.py --tool systems \
  --output-dir benchmarks/profiles/20260716T0505Z-systems          # 475MB QDSTRM；importer 失败
```

`compare_logits.py` 的最大 logits 绝对误差为 `0.0172214508`，低于固定
`atol=0.15, rtol=0.03`，32 个 greedy tokens 完全一致。正式一命令报告生成于远端
`benchmarks/results/20260716T045714Z/`，包含 JSON/CSV/Markdown；四个 context 均为 3 次完整
warmup、10 samples、32 output tokens，且报告校验通过 workload/checksum/SM/finite/memory matrix。

| Context | median TTFT ms | median TPOT ms | median decode tok/s | engine peak bytes |
|---:|---:|---:|---:|---:|
| 128 | 1062.686 | 8.105 | 123.393 | 2,225,078,272 |
| 512 | 4415.558 | 8.762 | 114.163 | 2,233,466,880 |
| 2048 | 19676.109 | 10.828 | 92.354 | 2,271,215,616 |
| 4096 | 45751.994 | 14.183 | 70.506 | 2,317,352,960 |

context=128 的工作负载温度范围为 52--72 C，达到并超过 90 tok/s 门槛。长上下文连续负载最高
87 C；现场查询在 2048/4096 阶段得到
`clocks_event_reasons.sw_thermal_slowdown=Active`，SM 时钟曾降到约 1545--1695 MHz。因此长上下文
吞吐只作为热态安全/分布记录，不接受为无降频性能基线。

M2 当前退出结论：

| M2 退出条件/审计项 | 状态 | 本轮证据或缺口 |
|---|---|---|
| 所有 FP16 kernel 与 reference 在容差内 | verified | CUDA test 233 checks；所有列明 kernel family 通过固定容差 |
| 全模型 logits 与 32-token greedy 对齐 | verified | 32,000 logits 最大误差 `0.0172214508`；32/32 tokens 完全一致 |
| context 128/512/2048/4096 无 NaN 和崩溃 | verified | 四 context 各 10 samples 完成；finite/report contract 全部通过 |
| context 128/512/2048/4096 无越界 | unverified | 普通运行无症状，但 Compute Sanitizer 未产生 memcheck 结果，不能据此证明无 OOB |
| Benchmark 报告可一条命令重建 | verified | 一条命令生成 JSON/CSV/Markdown 和完整四 context matrix |
| context=128 median FP16 decode >= 90 tok/s | verified | `123.393 tok/s`，10 samples，AC/Balanced，context-128 最高 72 C |
| CUDA skipped/disabled/0-tests/fallback | non-finding | inventory 7 enabled、0 disabled；CUDA test 实跑 1/1；无 skip/fallback marker |
| Compute Sanitizer | failed | 工具在首个 instrumented API 前终止；没有可接受的 memcheck 证据 |
| Nsight Systems / Compute | failed / unverified | Systems 仅留下原始 QDSTRM、importer 失败；主机随后离线，Compute 未运行 |
| thermal/clock review | failed | 2048/4096 明确观察到 software thermal slowdown；不接受长 context 性能为无降频基线 |
| 正式 benchmark 的 clean Git commit 基线 | failed | 本地和远端均为 `No commits yet`；任务不授权 commit，报告因此按设计 exit 2 |

因此 M2 仍为**未完成**，但数值正确性、四上下文有限值/崩溃检查、报告重建和 context-128 性能
门槛已有真实指定 GPU 证据。仍必须在目标主机恢复后取得 Compute Sanitizer 通过结果、可审阅的
Nsight Systems/Compute 报告，并解决获准的 clean commit 基线；所有必需项通过前不得开始 M3。

#### M1 只读再审计与 M2 固定 WSL 镜像实跑（2026-07-16 19:25 CST）

本轮以 Mac checkout 为唯一源码真源，固定 WSL 镜像为
`/home/xs/tinyllama-inference-engine`。每次 CUDA configure、build、test、sanitizer、profiling 和
benchmark 前均从 Mac 执行 `scripts/sync_to_wsl.sh`；远端没有直接编辑源码，也没有 Git
checkout。同步仍排除 `.git/`、`build/`、`models/`、generated data、benchmark 结果和 profile
产物。

M1 先完成只读再审计。仓库仍为 `No commits yet`、0 tracked files，65 个项目文件均未跟踪；
因此 `git diff` 和 `git diff --cached` 为空仍不构成历史基线。锁文件、格式、类型、Python tests、
CPU configure/build、CTest inventory、ASan 链接和 CLI smoke 全部重跑。CTest inventory 为 6
enabled、0 disabled；6/6 共 604 个显式 C++ checks 通过，14/14 Python tests 通过。当前测试日志
没有 skip、`Not Run`、0 tests、disabled 或 fallback marker。

| M1 退出条件/审计项 | 状态 | 本轮证据 |
|---|---|---|
| 固定 tokenizer 语料 token ID 完全一致 | verified | tokenizer 28 checks 通过 |
| FP32 关键 tensor/logits 在容差内 | verified | 113 traces、483 checks；最大绝对误差 `6.60419e-05` |
| Greedy 前 32 token 一致 | verified | integration regression 通过；smoke 首 token `3681` / `Paris` |
| 无 NVIDIA GPU 的 CPU tests 可运行 | verified | macOS arm64 Debug；CTest 6/6；unit/integration binary 均链接 ASan runtime |
| 可恢复失败路径 | verified | 非法 CLI 整数、checksum、shape、OOM、KV/tokenizer 边界 regression 均实跑 |
| skipped/disabled/0-tests/环境 fallback | non-finding | inventory 6 enabled、0 disabled；日志无相应 marker |

M1 没有新的必需缺口，因而继续且仅继续 M2。本轮 M2 实现/修复如下：

- 新增确定性 `.tlie-source-snapshot.json`：对 Mac 源码逐文件记录 size/SHA-256 和 tree SHA-256，
  同时保存 Mac Git commit/dirty metadata。WSL benchmark 在运行前后验证镜像与该快照完全一致；
  这取代了与“远端不复制 `.git`”规则冲突的远端 clean-commit gate，但不会伪造不存在的 commit。
- benchmark 增加 `clocks_event_reasons.sw_thermal_slowdown` 采集和合同校验；自动通过只覆盖吞吐与
  源码可复现前置条件，温度/时钟仍要求人工审阅。
- profiling 脚本现在要求生成非空 `.nsys-rep` / `.ncu-rep`。旧 Nsight Systems 在 importer
  失败时虽然返回 0、但只留下 QDSTRM 的假绿灯现已变为明确 exit 1；正常和失败路径均有 regression。
- CMake FetchContent 改为同一固定 revision 的 codeload tarball，并固定 tarball SHA-256，避免新
  WSL 镜像首次配置时下载完整 Git 历史；没有安装或升级任何全局软件、CUDA toolkit 或 driver。

Mac 实际验证：

```bash
mise exec -- uv lock --check                                      # 48 packages，锁文件有效
mise exec -- uv run ruff format --check scripts tests/*.py        # 14 files，格式通过
mise exec -- uv run ruff check scripts tests/*.py                 # 通过
mise exec -- uv run mypy scripts                                  # 12 files，无问题
mise exec -- uv run python -m unittest discover -s tests \
  -p 'test_*.py' -v                                               # 14/14，无 skip
xcrun clang-format --dry-run --Werror app/*.cpp app/*.cu \
  include/tlie/*.hpp include/tlie/cuda/*.hpp src/*.cpp src/cuda/* \
  tests/*.cpp tests/*.hpp tests/cuda/*.cu benchmarks/*.cu         # 通过
jq empty CMakePresets.json config/model_manifest.json \
  data/golden/*.json                                              # 通过
bash -n scripts/sync_to_wsl.sh                                    # 通过
cmake --preset cpu-debug && cmake --build --preset cpu-debug      # 配置/构建通过
ctest --test-dir build/cpu-debug --show-only=json-v1              # 6 enabled，0 disabled
ctest --preset cpu-debug --output-on-failure --verbose            # 6/6，34.39 秒
./build/cpu-debug/tinyllama_reference models/tinyllama-chat-v1.0 \
  "The capital of France is" 1 3681                              # token 3681，Paris
```

固定 WSL 镜像上的每组 CUDA 命令前均先单独运行 `scripts/sync_to_wsl.sh`。指定 GPU 环境为
16GB `NVIDIA GeForce RTX 3080 Laptop GPU`（SM 8.6），WSL2 kernel `6.18.33.1`、driver
`610.47`、CUDA toolkit `12.0.140`、runtime `12000`，Windows
`PowerOnline=true, Charging=false, Discharging=false`，电源计划为 Balanced。

```bash
scripts/sync_to_wsl.sh
ssh -p 2222 my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  cmake --preset cuda-release'                                    # 配置通过
scripts/sync_to_wsl.sh
ssh -p 2222 my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  cmake --build --preset cuda-release'                            # 全部 CUDA targets 构建通过
scripts/sync_to_wsl.sh
ssh -p 2222 my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  ctest --test-dir build/cuda-release -L cuda \
  --output-on-failure --verbose'                                  # 1/1；233 checks
scripts/sync_to_wsl.sh
ssh -p 2222 my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  UV_PYTHON_DOWNLOADS=never ~/.local/bin/uv run --offline \
  --python /usr/bin/python3 --no-project --with numpy==2.2.6 \
  python scripts/compare_logits.py'                               # 32/32 tokens，32,000 logits
scripts/sync_to_wsl.sh
ssh -p 2222 my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  /usr/bin/python3 scripts/benchmark.py \
  --power-mode "AC/Balanced (PowerOnline=true, Charging=false, \
  Discharging=false)"'                                           # exit 0；JSON/CSV/Markdown
```

CUDA CTest inventory 为 7 enabled、0 disabled；Mac 上的 6 个 CPU tests 与 WSL 上 label `cuda`
的 1 个真实 GPU test 全部运行，没有 skip、`Not Run`、0-tests 或 backend fallback。CUDA kernel
test 在真实 RTX 3080 Laptop 上完成 233 checks。`compare_logits.py` 的 32 个 greedy tokens
完全一致，32,000 logits 最大绝对误差 `0.0172214508`，低于固定
`atol=0.15, rtol=0.03`。

最终正式报告生成于 WSL
`benchmarks/results/20260716T112501Z/`。它记录 65 文件源码树在运行前后均为
`a61d50597ef2270731de12106b90cabe98a5ef8b71ad6de2cf93a0aa3b4f5821`；Mac Git metadata
诚实记录为 `commit: null, working_tree_dirty: true`。每个 context 都完成 3 次完整 warmup、10 个
samples 和 32 个 output tokens。

| Context | median TTFT ms | median TPOT ms | median decode tok/s | engine peak bytes |
|---:|---:|---:|---:|---:|
| 128 | 1188.266 | 9.167 | 109.097 | 2,225,078,272 |
| 512 | 4663.010 | 9.328 | 107.369 | 2,233,466,880 |
| 2048 | 23197.728 | 12.809 | 78.069 | 2,271,215,616 |
| 4096 | 53916.443 | 17.111 | 58.441 | 2,317,352,960 |

context=128 的监控覆盖 33 个样本，59--81 C，software thermal slowdown 始终
`Not Active`，因此 `109.097 tok/s` 可用于 90 tok/s 门槛。512/2048/4096 均观察到
software thermal slowdown `Active`，最高 87 C；这些长上下文数字只作为有限值/崩溃/热态分布
记录，不接受为无降频性能基线。

Compute Sanitizer 在每次尝试前均重新同步源码。显式 injection path、`target-processes=all` 和 UDS
连接覆盖的尝试都在首个 instrumented API 前终止（exit 255）；不指定 injection path 则找不到
`libsanitizer-collection.so`（exit 13）。目标和 injection library 的 `ldd` 均无缺失依赖，普通
CUDA test 同环境通过，因此这是 CUDA 12.0 / Compute Sanitizer 2022.4.1 的 WSL 工具启动失败，
不是可接受的 memcheck 结果。没有未经授权升级 toolkit/driver。

Nsight Systems 2022.4.2 能捕获 CUDA API（例如 `cudaLaunchKernel` 107,460 calls），但自动 importer
失败；手动导入得到 `.nsys-rep` 后，`gpukernsum` 仍明确报告没有 CUDA kernel data。修复后的脚本
已把仅 QDSTRM 的情况变为 exit 1。Nsight Compute 2022.4.1 连接目标后因
`ERR_NVGPUCTRPERM` 退出 255，并明确报告 `No kernels were profiled`；没有未经授权修改系统性能
计数器权限。

M2 当前退出结论：

| M2 退出条件/审计项 | 状态 | 本轮证据或缺口 |
|---|---|---|
| 所有 FP16 kernel 与 reference 在容差内 | verified | 真实 GPU CUDA test 233 checks；列明 kernel family 和边界均通过 |
| 全模型 logits 与 32-token greedy 对齐 | verified | 32,000 logits 最大误差 `0.0172214508`；32/32 tokens 一致 |
| context 128/512/2048/4096 无 NaN 和崩溃 | verified | 每个 context 3 warmups + 10 samples；finite/report contract 通过 |
| context 128/512/2048/4096 无越界 | unverified | 普通执行和边界 tests 无症状，但 Compute Sanitizer 没有产生 memcheck 证据 |
| Benchmark 报告可一条命令重建 | verified | 单命令 exit 0；非空 JSON/CSV/Markdown；源码快照前后一致 |
| context=128 median FP16 decode >= 90 tok/s | verified | `109.097 tok/s`；10 samples；该阶段 software thermal slowdown 始终 Not Active |
| CUDA skipped/disabled/0-tests/fallback | non-finding | 7 enabled、0 disabled；CUDA 1/1 实跑；无 skip/Not Run/fallback |
| Compute Sanitizer | failed | 工具在目标首个 instrumented API 前失败；无 memcheck 报告 |
| Nsight Systems kernel timeline | failed | API trace 存在，但导入报告明确为 0 GPU kernel/memory events |
| Nsight Compute metrics | failed | `ERR_NVGPUCTRPERM`；0 kernels profiled |
| thermal/clock review | verified / failed | context=128 无 thermal slowdown；512/2048/4096 有 slowdown，长上下文性能基线不接受 |
| 可复现源码基线 | verified | Mac 65 文件 tree SHA-256 前后一致；不存在的 Git commit 仍明确记录为 null |

因此 M2 仍为**未完成**，没有开始 M3。四项显式退出条件中，“无越界”仍为 `unverified`；此外
本里程碑要求的可用 Nsight Systems kernel timeline 和 Nsight Compute metrics 均未取得。继续完成
M2 需要用户授权后升级/更换兼容的 CUDA profiling/sanitizer 工具环境，或授权管理员启用 GPU
performance counters；在 Compute Sanitizer 真实通过且 profiler 产生非零 kernel 数据前不得标记
完成。

#### M1 最终复审与 M2 恢复验收（2026-07-16 23:27 CST）

**M2 已完成。** 本段取代上方“工具不可用”时的临时结论；只完成 M2，M3 没有开始。

本轮先完整读取当前 `AGENTS.md`、`PLAN.md`、69 文件源码快照、测试、Git 状态、普通/缓存
diff 和既有验证记录，再复验 M1。仓库仍是 `No commits yet`、0 tracked files；因此
`git diff`/`git diff --cached` 为空只是没有 tracked baseline，不是“无改动”证据。Mac 最终重建
`cpu-debug`，CTest inventory 为 6 enabled、0 disabled；CTest 6/6 共 604 个显式 C++ checks，
Python 22/22，均无 skip、`Not Run`、0-tests 或 environment fallback。CLI smoke 仍为 token
`3681` / `Paris`。

| M1 退出条件/审计项 | 状态 | 最终证据 |
|---|---|---|
| 固定 tokenizer 语料 token ID 完全一致 | verified | tokenizer 28 checks；固定 7-case corpus |
| FP32 关键 tensor/logits 在容差内 | verified | 113 traces、483 integration checks；最大绝对误差 `6.60419e-05` |
| Greedy 前 32 token 一致 | verified | integration regression 通过；CLI smoke `3681` / `Paris` |
| 无 NVIDIA GPU 的 CPU tests 可运行 | verified | macOS arm64 Debug；ASan/UBSan preset；CTest 6/6 |
| 可恢复失败路径 | verified | checksum、shape、OOM、KV/tokenizer 边界和非法 CLI 整数 regression 均实跑 |
| skipped/disabled/0-tests/环境 fallback | non-finding | CTest 6 enabled、0 disabled；Python 22/22；日志无 skip/Not Run/fallback |

M1 没有 `failed` 或 `unverified` 的必需退出项，因而允许恢复且仅恢复 M2。本轮针对此前真实失败
补齐以下实现和环境修复：

- `scripts/wsl_cuda_env.py` 集中构造 WSL CUDA target 环境：固定当前 WSL driver 目录和
  `/usr/lib/wsl/lib`，检测冲突的 Linux `libcuda.so.1`，只对 CUDA target 注入由
  `scripts/wsl_cuda_ld_audit.c` 构建的 `LD_AUDIT` 映射，避免修改全局 loader 状态。
- `scripts/run_cuda_memcheck.py` 只接受同时包含 clean leak summary 和 `ERROR SUMMARY: 0 errors`
  的真实 memcheck 日志；`scripts/profile_cuda.py` 只接受非空、可导入的 `.nsys-rep`/`.ncu-rep`、
  非零 kernel rows 以及 `prefill`/`decode`/`sampling` NVTX ranges。
- `scripts/benchmark.py` 原先直接启动 CUDA 子进程，遗漏同一 WSL loader 环境，重启后首次正式
  benchmark 因而在 microbenchmark 阶段以 `no CUDA-capable device is detected` 失败。现已让所有
  kernel/engine 子进程统一经 `CudaToolEnvironment` 启动，并增加 regression 证明包装命令和环境
  实际传给 `subprocess.Popen`；修复后完整 benchmark 通过。
- Windows 端按用户授权启用 debugger 接口注册表设置、NVIDIA performance counter “All users”
  权限，并重启 Windows。重启后 driver log 为 `isNonAdminPerfCountAccessAllowed:1`；当前 NVIDIA
  adapter、`nvlddmkm` global 和 video adapter 的 `RmProfilingAdminOnly` 均为 `0`。没有升级或替换
  driver/CUDA toolkit。
- 新版 sanitizer/Nsight 只用 `dpkg-deb -x` 解压到 WSL 用户目录
  `~/.local/opt/tlie-cuda-tools/`，未全局安装。工具版本和删除方式记录在“工具安装记录”。

Mac 最终非 CUDA 验证：

```bash
mise exec -- uv lock --check                                      # 48 packages
mise exec -- uv run ruff format --check scripts tests/*.py        # 17 files
mise exec -- uv run ruff check scripts tests/*.py                 # 通过
mise exec -- uv run mypy scripts                                  # 14 files，无问题
mise exec -- uv run python -m unittest discover -s tests \
  -p 'test_*.py' -v                                               # 22/22，无 skip
xcrun clang-format --dry-run --Werror app/*.cpp app/*.cu \
  include/tlie/*.hpp include/tlie/cuda/*.hpp src/*.cpp src/cuda/* \
  tests/*.cpp tests/*.hpp tests/cuda/*.cu benchmarks/*.cu \
  scripts/wsl_cuda_ld_audit.c                                     # 通过
jq empty CMakePresets.json config/model_manifest.json \
  data/golden/*.json benchmarks/results/20260716T151538Z/report.json # 通过
bash -n scripts/sync_to_wsl.sh                                    # 通过
cmake --preset cpu-debug && cmake --build --preset cpu-debug      # 配置/构建通过
ctest --test-dir build/cpu-debug --show-only=json-v1              # 6 enabled，0 disabled
ctest --preset cpu-debug --output-on-failure --verbose            # 6/6；31.63 秒
./build/cpu-debug/tinyllama_reference models/tinyllama-chat-v1.0 \
  "The capital of France is" 1 3681                              # token 3681；Paris
```

Mac checkout 继续是唯一源码真源；以下每个 CUDA configure/build/test/sanitizer/profile/benchmark
命令前均单独执行 `./scripts/sync_to_wsl.sh`，目标为
`my-wsl:~/tinyllama-inference-engine`（实际目录 `/home/xs/tinyllama-inference-engine`）。所有最终
CUDA 证据对应 69 文件可执行源码快照
`fbbbe386f92c07c3dc9e0cdb24f7ca1b64f6010f0b9df3779ce3fe1941735ec4`；本段 PLAN 验证记录是在
所有测试通过后按规则追加的文档变更，不改变可执行代码。

指定环境为 16GB `NVIDIA GeForce RTX 3080 Laptop GPU`（SM 8.6）、driver `610.47`、WSL2
kernel `6.18.33.1-microsoft-standard-WSL2`、CUDA toolkit `12.0.140`、runtime `12000`。Windows
状态为 `PowerOnline=true, Charging=false, Discharging=false`，电源计划 Balanced。

最终 GPU 命令与结果（每段前的 source sync 均返回同一个 69-file tree hash）：

下列 CUDA target argv 与实际运行一致；CTest 和 logits target 由 `CudaToolEnvironment.wrap_target`
启动，sanitizer/profile/benchmark 脚本在内部使用同一包装器。为可读性省略了只负责定位已安装
工具的 `PATH` 前缀。

```bash
./scripts/sync_to_wsl.sh
ssh my-wsl 'cd ~/tinyllama-inference-engine && cmake --preset cuda-release'
                                                                    # 配置通过
./scripts/sync_to_wsl.sh
ssh my-wsl 'cd ~/tinyllama-inference-engine && cmake --build --preset cuda-release'
                                                                    # 全部 targets 通过
./scripts/sync_to_wsl.sh
ssh my-wsl 'cd ~/tinyllama-inference-engine && \
  ctest --test-dir build/cuda-release -L cuda --output-on-failure --verbose'
                                                                    # 1/1；233 checks
./scripts/sync_to_wsl.sh
ssh my-wsl 'cd ~/tinyllama-inference-engine && \
  UV_PYTHON_DOWNLOADS=never ~/.local/bin/uv run --offline \
  --python /usr/bin/python3 --no-project --with numpy==2.2.6 \
  python scripts/compare_logits.py'                                 # 32/32 tokens；32,000 logits
./scripts/sync_to_wsl.sh
ssh my-wsl 'cd ~/tinyllama-inference-engine && \
  python3 scripts/run_cuda_memcheck.py \
  --output-dir benchmarks/profiles/20260716T1530Z-memcheck-fbb'     # 233 checks；0 leak；0 errors
./scripts/sync_to_wsl.sh
ssh my-wsl 'cd ~/tinyllama-inference-engine && \
  python3 scripts/profile_cuda.py --tool both \
  --output-dir benchmarks/profiles/20260716T1532Z-both-fbb'         # Systems + Compute exit 0
./scripts/sync_to_wsl.sh
ssh my-wsl 'cd ~/tinyllama-inference-engine && \
  python3 scripts/benchmark.py --contexts 128 512 2048 4096 \
  --batch 1 --power-mode \
  "AC/Balanced (PowerOnline=true, Charging=false, Discharging=false)"'
                                                                    # 自动验收 PASS
```

CUDA CTest inventory 为 7 enabled、0 disabled，label `cuda` 的 1 个测试在 RTX 3080 Laptop
实际完成 233 checks；没有 skip、`Not Run`、0-tests、disabled 或 CPU/backend fallback。
`compare_logits.py` 的 32/32 greedy tokens 完全一致，32,000 logits 最大绝对误差
`0.017221450805664062`，低于固定 `atol=0.15, rtol=0.03`。

Compute Sanitizer `2026.2.1.0` 的最终日志在
`benchmarks/profiles/20260716T1530Z-memcheck-fbb/compute-sanitizer.log`，包含
`LEAK SUMMARY: 0 bytes leaked in 0 allocations` 和 `ERROR SUMMARY: 0 errors`；目标测试同时完成 233
checks。因此四个 context 的普通/边界执行与 selected kernel memcheck 共同支持“无越界”，不再把
仅普通执行的无症状结果冒充 sanitizer 证据。

Nsight Systems `2026.1.3.425` 生成 8,848,636-byte `.nsys-rep`、11 个非零 kernel summary rows，
并捕获 `warmup`、`prefill`、`decode`、`sampling` 四个 NVTX ranges。Nsight Compute
`2026.2.1.0` 生成 36,904,134-byte `.ncu-rep` 和 276,454-byte raw CSV；实际 profile 12 个 launch、
每个 45 passes，得到 12 rows/9 个唯一 kernel names，覆盖 custom RMSNorm、RoPE、Softmax、KV、
Attention、SiLU、residual 和 cuBLAS GEMV/GEMM。没有 `ERR_NVGPUCTRPERM`、0-kernel 或仅 API/QDSTRM
假绿。

最终 benchmark 报告为 `benchmarks/results/20260716T151538Z/`。tree hash 运行前后均为上述
`fbbbe386...`，Git metadata 诚实记录 `commit: null, working_tree_dirty: true`。每个 context 均为
batch 1、32 output tokens、3 次完整 warmup、10 个真实 samples，全部测量字段为 finite；报告记录
固定模型 source/FP16 checksums、GPU/VRAM/driver/CUDA、功耗状态和工作负载期 GPU 监控。

| Context | median TTFT ms | median TPOT ms | median decode tok/s | thermal review |
|---:|---:|---:|---:|---|
| 128 | 1129.876 | 9.037 | 110.653 | 58--79 C；slowdown 始终 `Not Active`，可接受为门槛基线 |
| 512 | 4813.271 | 9.797 | 102.088 | 62--86 C；出现 slowdown，只作为受热限制分布 |
| 2048 | 21525.622 | 11.868 | 84.268 | 66--87 C；出现 slowdown，只作为受热限制分布 |
| 4096 | 49865.145 | 15.642 | 63.933 | 70--87 C；出现 slowdown，只作为 extrapolated stress 结果 |

128 阶段 31 个一秒监控样本均无 software thermal slowdown，故 `110.653 tok/s` 可用于 M2 的
`>= 90 tok/s` 固定门槛。其余 context 的有限值、无崩溃和显存边界仍有效，但由于观察到热节流，
不接受为“无降频性能”结论；4096 继续仅是超过训练长度的 RoPE extrapolation stress，不是质量
声明。

| M2 退出条件/审计项 | 状态 | 最终证据 |
|---|---|---|
| 所有 FP16 kernel 与 PyTorch/cuBLAS reference 在容差内 | verified | 真实 GPU 233 checks；12-row/9-family microbenchmark matrix；32,000-logit 对齐 |
| context 128/512/2048/4096 无 NaN 和崩溃 | verified | 每个 context 3 warmups + 10 samples；全部数值 finite |
| context 128/512/2048/4096 无越界 | verified | 4096 边界 tests + Compute Sanitizer 233 checks、0 errors |
| Benchmark 报告可一条命令重建 | verified | 单命令生成非空 JSON/CSV/Markdown；source hash 前后一致 |
| 固定条件 FP16 decode >= 90 tok/s | verified | context=128 median `110.653 tok/s`；10 samples；无 thermal slowdown |
| CUDA skipped/disabled/0-tests/fallback | non-finding | inventory 7 enabled、0 disabled；CUDA 1/1 实跑 |
| Compute Sanitizer | verified | 2026.2.1.0；0 leak、0 errors；非空日志 |
| Nsight Systems kernel timeline | verified | 可导入 8.85 MB report；11 kernel rows；4 NVTX ranges |
| Nsight Compute metrics | verified | 可导入 36.90 MB report；12 launches/rows；9 unique kernels |
| thermal/clock review | verified / qualified | 128 无节流；长 context 节流已明确限定，不作无节流性能宣称 |
| 可复现源码基线 | verified | 69 文件 tree hash 前后一致；不存在的 commit 明确记录为 null |

结论：M2 四项显式退出条件及里程碑要求的 sanitizer/profiling 证据全部 `verified`，环境 gating
审计为 `non-finding`；M2 标记完成。本轮在此停止，未实现 INT8、融合、KV 优化或任何 M3 内容。

### M3：INT8、融合与显存优化

#### 工作内容

- 实现 INT8 weight-only 和 per-channel scale。
- 记录量化、反量化和内存布局。
- 尝试 fused RMSNorm+residual、RoPE+QK transform 或其他 profiler 支持的融合。
- 对比 FP16、INT8 的 logits、困惑度代理、生成结果、tok/s 和显存。
- 优化 KV Cache 布局和预分配。
- 所有优化先建立 regression test，再改实现。

#### 退出条件

- INT8 不出现明显生成退化，误差报告完整。
- INT8 相对 FP16 吞吐提升至少 20%，或峰值显存下降至少 25%。
- 若未达阈值，必须通过 profiler 给出可重复的 non-finding，不能声称加速。
- 长上下文期间 KV Cache 无泄漏和重复分配。

#### M2 最终复审与 M3 验收（2026-07-19 CST）

**M3 已完成。** 本轮先只读复审 M2，修复退出条件相关缺口并重跑同一验收栈；M2 全部必需项
通过后才开始 M3。只实现 M3，未开始 M4 的 scheduler、batching、server 或 API。

仓库仍是 `No commits yet`、0 tracked files；因此空的普通/缓存 `git diff` 不能证明工作树无改动。
可复现性继续使用 Mac 端逐文件 source snapshot。M2 复审发现并修复两个真实缺口：

- `compare_logits.py` 原先直接启动 CUDA engine，绕过 WSL 冲突 `libcuda` 的 target wrapper；现统一
  使用 `CudaToolEnvironment` 并增加 subprocess regression。
- `benchmark.py` 原先要求 `nvidia-smi` 在普通 `PATH`，非交互 WSL 会错误 gating；现允许已验证的
  `/usr/lib/wsl/lib/nvidia-smi` 标准映射并增加 regression。

修复后的 M2 可执行源码快照为
`8608106d...`。Mac CPU CTest 为 6/6、604 checks；Python 当时为 30/30；RTX 3080 CUDA CTest
为 7 enabled、0 disabled，CUDA test 233 checks。最终 logits 对齐为 32/32 greedy tokens、32,000
logits、最大绝对误差 `0.0172214508`。Compute Sanitizer 为 233 checks、0 leak、0 errors；Nsight
Systems/Compute 均生成可导入报告，Compute 捕获 12 launches、每个 45 passes。M2 正式 benchmark
覆盖 128/512/2048/4096、3 warmups、10 samples，128 decode 为 `107.903 tok/s`；长 context 的热
节流只作为 qualified stress 证据。

| M2 退出条件/审计项 | 状态 | 本轮复审证据 |
|---|---|---|
| FP16 kernel/reference 与 logits/token 对齐 | verified | 233 CUDA checks；32/32 tokens；32,000 logits |
| 128/512/2048/4096 finite、无崩溃、无越界 | verified | 全矩阵 benchmark；selected-kernel memcheck 0 errors |
| 固定条件 FP16 decode >= 90 tok/s | verified | context 128 median `107.903 tok/s` |
| benchmark/profiler 可复现 | verified | source hash 前后一致；非空 JSON/CSV/Markdown、Nsight reports |
| skipped/disabled/0-tests/backend fallback | non-finding | 7 enabled、0 disabled；CUDA label 1/1；无 skip/Not Run/fallback |
| WSL engine wrapper 与 `nvidia-smi` gating | failed -> repaired | 两项 regression + 同一真实 GPU 验收栈通过 |
| 必需退出条件剩余缺口 | non-finding | 无 `failed` 或 `unverified` 必需项；允许进入 M3 |

M3 实现 W8A16 symmetric per-output-channel：156 个 rank-2 matrix 使用每输出行一个 FP32 scale，
rank-1 RMSNorm 保持 FP16。混合 TLIEWGT 共 357 records，其中 201 个 model tensors；严格重建得到
`1,102,019,200` bytes、SHA-256
`5e394d62994202b4a9a66dadc48b0d775341050645c1f61c37c6e4bbf876a422`。loader 在上传前校验外部
完整文件/source pins、record dtype/rank/shape/byte count、tensor digest 与 scale shape。CUDA runtime
新增 W8A16 GEMV 和 embedding-row dequantization；activation、RMSNorm、KV Cache 与 workspace 保持
FP16。格式、kernel contract 和验收边界记录于
`docs/adr/0003-m3-int8-weight-only.md`。

M3 新增 regression 后的最终非 CUDA验证：

```bash
uv lock --check                                                   # 48 packages
uv run ruff format --check scripts tests                          # 24 files
uv run ruff check scripts tests                                   # 通过
uv run mypy scripts                                               # 17 source files，无问题
uv run python -m unittest discover -s tests -p 'test_*.py' -v    # 30/30，无 skip
xcrun clang-format --dry-run --Werror app/* include/tlie/* \
  src/* tests/* benchmarks/*.cu scripts/wsl_cuda_ld_audit.c       # 通过
jq empty CMakePresets.json config/model_manifest.json \
  data/golden/*.json                                              # 通过
bash -n scripts/sync_to_wsl.sh                                    # 通过
cmake --preset cpu-debug && cmake --build --preset cpu-debug      # 通过
ctest --preset cpu-debug --output-on-failure --verbose            # 6/6；604 checks
./build/cpu-debug/tinyllama_reference models/tinyllama-chat-v1.0 \
  "The capital of France is" 1 3681                              # token 3681；Paris
```

以下每个 CUDA build/test/sanitizer/profile/benchmark 前均从 Mac 单独运行
`TLIE_WSL_DIR=tinyllama-inference-engine ./scripts/sync_to_wsl.sh`，随后只在
`/home/xs/tinyllama-inference-engine` 构建或执行。源码 sync 排除大模型；一次全量 CUDA CTest 因远端
FP32 artifact 只有 `.partial` 而失败，补齐并核验只读模型资产后，同一 7-test CTest 全部通过。一次
绕过 wrapper 的手工 microbenchmark 因冲突 Linux `libcuda` 报 `no CUDA-capable device`，该结果作废；
正式命令均经 repository wrapper 重跑。没有修改 WSL source、driver、toolkit 或全局依赖。

```bash
# 每段前均执行上述 Mac source sync
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  cmake --preset cuda-release && cmake --build --preset cuda-release'
                                                                    # 全部 targets 通过
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  <CudaToolEnvironment> ctest --preset cuda-release \
  --output-on-failure --verbose'                                    # 7/7；CUDA 249 checks
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python scripts/compare_int8.py \
  --output benchmarks/results/20260719T-m3-int8/accuracy-final-0917.json'
                                                                    # accuracy PASS
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python scripts/run_cuda_memcheck.py \
  --output-dir benchmarks/profiles/20260719T-m3-memcheck-7a4b'      # 249 checks；0 leak/errors
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python scripts/profile_cuda.py --tool both --mode int8 \
  --output-dir benchmarks/profiles/20260719T-m3-int8-profile-0917'  # Systems + Compute PASS
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python scripts/benchmark_int8.py --contexts 128 4096 \
  --output-tokens 32 --warmup 3 --samples 10 \
  --power-mode "AC/Balanced (PowerOnline=true, Charging=false, Discharging=false)" \
  --output-dir benchmarks/results/20260719T-m3-final-af4f'          # threshold PASS
```

最终 correctness/sanitizer/profiler/microbenchmark 使用 77-file source snapshot
`0917250637d853baa72b5ae70acb48310062c68ff58f8de591d617ae40231853`。正式 paired benchmark
使用 production-engine snapshot
`af4fe4d3c6637708b0c402ee9f2be5532498484d86c93c76dbbc6031b7121068`，运行前后 hash 完全一致；
后续差异只增加 microbenchmark/profile coverage、tests 和文档，没有修改该次被测的 model/loader/
kernel/engine 路径。本段 PLAN 更新发生在所有验证之后，不属于被测 executable snapshot。

Accuracy 报告同时比较 FP32、FP16、INT8：INT8 对 FP32 的 max/mean/RMS absolute error 为
`0.299961` / `0.049828` / `0.063210`，cosine `0.9999246`，top-1 token `3681` 一致，JS divergence
`0.00013709`，top-1 perplexity ratio proxy `0.988456`。INT8 greedy 与 golden 公共前缀 11/32、位置
一致率 71.875%；文本仍连贯，但不是 FP16 exact-token equivalence，此差异已在报告中保留而未隐藏。

正式 paired benchmark 为 batch 1、greedy、32 output tokens、3 个 full-workload warmups、10 个
samples；所有计时值 finite，模型 hash 和 source snapshot 匹配：

| Mode | Context | median TTFT ms | median TPOT ms | median decode tok/s | peak device bytes | KV bytes | allocations before/after |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 128 | 1108.024 | 8.921 | 112.092 | 2,225,078,272 | 3,604,480 | 214 / 214 |
| INT8 | 128 | 1231.195 | 9.755 | 102.513 | 1,189,085,184 | 3,604,480 | 370 / 370 |
| FP16 | 4096 | 47770.242 | 14.651 | 68.257 | 2,317,352,960 | 92,995,584 | 214 / 214 |
| INT8 | 4096 | 57317.188 | 17.498 | 57.149 | 1,283,457,024 | 92,995,584 | 370 / 370 |

context 128 的峰值 device memory 下降 `46.560%`，超过 `>=25%` 门槛；decode throughput 下降
`8.546%`，因此是明确的 speed non-finding，不能声称 INT8 加速。128 的 FP16/INT8 workload 分别
有 32/26 个一秒监控样本，software thermal slowdown 始终 `Not Active`；4096 两条路径均出现热
节流，只用于长上下文 safety/memory evidence，不作为无节流性能结论。

无 profiler 扰动的同 shape microbenchmark 中，cuBLAS FP16 GEMV median `0.065536 ms`，当前 INT8
GEMV `0.099328 ms`，局部慢 `51.6%`。Nsight Systems 的真实 INT8 run 中，W8A16 GEMV 占 kernel
时间 `92.0%`，RMSNorm `3.1%`、residual `0.9%`、embedding dequantization约 `0.02%`；因此此时做
RMSNorm+residual 或 embedding fusion 不能解决主瓶颈，保留为 profiler-backed non-finding。Nsight
Compute 生成 40,319,533-byte report，实际 profile 14 launches、每个 45 passes，包含
`Int8WeightOnlyGemvKernel` 与 `Int8EmbeddingRowKernel`；Systems 生成 5,521,329-byte report，并
捕获 `warmup`、`prefill`、`decode`、`sampling` NVTX ranges。未来若追求速度，应先替换当前 scalar
FP32-reduction GEMV，而不是提前堆叠低占比 fusion。

KV Cache 继续按 declared maximum 在 model create 时一次预分配；FP16/INT8 在同 context 的 KV
bytes 完全相同，四个 128/4096 workload 的 device allocation count 均前后不变。Compute Sanitizer
`2026.2.1.0` 对包含两个新 INT8 kernel 的 249 checks 报 `0 bytes leaked`、`ERROR SUMMARY: 0
errors`。没有证据支持为 M3 改变已经满足 ownership/bounds/preallocation contract 的 KV layout，
故未加入无 profiler 支持的布局复杂度。

| M3 退出条件/审计项 | 状态 | 最终证据 |
|---|---|---|
| INT8 无明显生成退化且误差报告完整 | verified / qualified | top-1 一致、cosine 0.9999246、prefix 11；token drift 明确披露 |
| 吞吐 +20% 或峰值显存 -25% | verified (memory only) | memory -46.560%；throughput -8.546% |
| 未加速时 profiler-backed non-finding | verified | real INT8 Systems：GEMV 92.0%；Compute 14 launches；microbench 慢 51.6% |
| 长上下文 KV 无泄漏和重复分配 | verified | 4096 allocation counts 稳定；KV bytes 一致；memcheck 0 leak/errors |
| INT8 format/loader/kernel contracts | verified | deterministic SHA；strict loader checks；Python 30/30；CUDA 249 checks |
| skipped/disabled/0-tests/fallback | non-finding | Mac 6、WSL 7 enabled；0 disabled；无 skip/Not Run/0-tests/backend fallback |
| fusion 和 KV layout 改写 | non-finding | profiler 显示候选 fusion 低占比；现有 KV 预分配已满足退出条件，未做无证据改写 |
| 必需退出条件剩余缺口 | non-finding | 无 `failed` 或 `unverified` 必需项 |

结论：M3 四项显式退出条件全部通过；收益是显存而非速度，速度 non-finding 有可重复 profiler
证据。M3 标记完成，本轮停止；M4 保持未开始。

### M4：批处理调度与服务化

#### 工作内容

- 设计请求队列和生命周期：queued/running/completed/cancelled/failed。
- 实现简化 continuous batching。
- 分离 prefill 和 decode 调度。
- 建立 KV Cache block allocator，并在取消/失败时回收。
- 提供 `/v1/models` 和 `/v1/chat/completions`。
- 支持 SSE、客户端取消、timeout、max_tokens、temperature、top-p。
- 暴露 TTFT、TPOT、tok/s、queue time、active sequences、KV usage。

#### 退出条件

- OpenAI-compatible smoke client 可完成流式和非流式生成。
- 20 个并发请求均可完成、取消或超时，无悬挂请求。
- Batch 4 总吞吐达到 Batch 1 的至少 1.5 倍。
- 取消请求后对应 KV block 被回收。

#### M3 复审、缺口修复与 M4 验收记录（2026-07-19 CST）

状态：M3 的必需缺口已修复并重新验证；M4 已完成。本节记录形成时 M5 尚未开始。本轮所有 CUDA
configure/build/test/Sanitizer/benchmark/smoke 前均从 Mac 运行
`TLIE_WSL_DIR=tinyllama-inference-engine ./scripts/sync_to_wsl.sh`，WSL 镜像固定为
`/home/xs/tinyllama-inference-engine`，没有在 WSL 修改或反向复制源码。

M3 只读复审首先重现了旧 INT8 artifact 的准确性结果。旧报告虽满足数值阈值，但生成文本从
FP16 的 `Germany is Berlin ... United States is Washington` 退化为重复的
`France is Paris`，所以旧的“无明显退化”退出条件判定为 **failed**，不是通过。修复为 hybrid
W8A16：154 个内部 rank-2 矩阵按输出通道量化，`embed_tokens` 与 `lm_head` 两个词表矩阵保留
FP16；同时把准确性 gate 收紧为 32/32 golden token 完全一致。

| M3 退出条件/审计项 | 结论 | 本轮证据 |
|---|---|---|
| 32-token greedy 与 FP16/golden，无明显文本退化 | verified（修复后） | `benchmarks/results/20260719T-m3-hybrid-reaudit/accuracy.json`：FP16/INT8 都为 32/32 exact，cosine `0.99994158`；旧 artifact 复审为 failed |
| INT8 吞吐提升 >=20% 或显存下降 >=25% | verified | `benchmarks/results/20260719T-m3-hybrid-final/report.json`：context 128 峰值显存下降 `40.434%`；吞吐 `-5.108%` 为 non-finding，不冒充加速 |
| profiler 指向 INT8 kernel/带宽瓶颈 | verified | `benchmarks/profiles/20260719T-m3-hybrid-profile/`：INT8 GEMV 约占 Systems kernel 时间 `88%`；保留 FP16 lm_head 的 cuBLAS 约 `4%` |
| KV/分配稳定且 selected CUDA memory path 安全 | verified | context 128/4096 的前后 allocation count 分别保持 `214/214`、`368/368`，KV bytes 相同；`20260719T-m3-hybrid-memcheck-final` 为 249 checks、0 leak、0 errors |
| skipped、disabled、0 tests、fallback/gating | non-finding | CPU 6/6、Python 30/30、CUDA 7/7；未见 skip/disabled/0-tests/backend fallback。无 CMake PATH、无 NumPy、错误 sanitizer 注入库的尝试均硬失败并在正确环境重跑，未计作通过 |

M4 实现单份共享权重、4-row CUDA batch forward、独立 slot-major KV Cache、可回收的 20-slot
allocator、decode-priority rotating continuous batching，以及 loopback OpenAI-compatible 服务。
请求状态为 queued/running/completed/cancelled/failed；支持非流式、SSE、disconnect cancellation、
timeout、max_tokens、temperature/top-p，并暴露 queue、TTFT、TPOT、tok/s、active sequence 与 KV
利用率。协议和调度决策记录于 `docs/adr/0004-m4-batching-server.md`。cpp-httplib 固定到不可变提交
`5814e121dfb5049f72a5c3956c3c8961b40da78b`。

最终 pre-PLAN 被测源码为 82-file snapshot
`72ae7569154bc6dbe82455f5ac5d006428220a7c0d7a7043e01eb61bb20d89c9`；commit 为 `null`，
working tree dirty（仓库当前没有 tracked files）。正式 Batch 报告位于
`benchmarks/results/20260719T-m4-final-72ae/batch.json`：RTX 3080 Laptop 16 GiB、driver
`610.47`、CUDA toolkit/runtime `12.0/12000`、FP16 model SHA
`1038f532a16e316f1953cf721cb2a54783cd1327b6a22e45ce71dcc2c574ab63`、context 128、output
32、3 warmups、10 samples。Batch 1/4 总吞吐分别为 `21.636/63.171 tok/s`，比值
`2.920x`，各 row token 完全一致。55 个 workload monitor 样本记录 48--74 C、SM 210--1800
MHz、P0/P8，software thermal slowdown 始终 `Not Active`；低时钟包含采样间隙，未选择性剔除。

| M4 退出条件/审计项 | 结论 | 本轮证据 |
|---|---|---|
| OpenAI-compatible 流式和非流式 smoke | verified | `benchmarks/results/20260719T-m4-final-72ae/smoke.json`：models、nonstream、SSE、temperature/top-p、稳定 error shape 全部通过 |
| 20 并发均完成、取消或超时，无悬挂 | verified | 12 completed、4 timeout、4 client-cancel；最终 queued/active 为 0，failed 为 0 |
| Batch 4 总吞吐 >= Batch 1 的 1.5 倍 | verified | 正式 10-sample 报告为 `2.920x`；`tokens_match=true` |
| 取消/超时后 KV block 回收 | verified | smoke 结束 `kv_blocks_used=0`、`kv_utilization=0`，8 个 server-side cancelled terminal states 均释放 slot |
| CUDA batch memory safety | verified | `benchmarks/profiles/20260719T-m4-final-72ae-memcheck/compute-sanitizer.log`：真实 batch target，0 bytes leaked、0 errors |
| skipped、disabled、0 tests、environment gating、fallback | non-finding | CUDA inventory 为 8 enabled、0 disabled；完整 CUDA CTest 8/8（kernel 249 checks），最终 batch CTest 1/1；无 skip/Not Run/0-tests/CPU fallback。系统 CUDA 12 sanitizer 的两种失败均未采信，改用既有且校验过的 CUDA Sanitizer 13.3 用户缓存解压件后重跑通过 |

实际验收命令摘要：

```bash
# 以下每个 ssh CUDA 段前均单独执行
TLIE_WSL_DIR=tinyllama-inference-engine ./scripts/sync_to_wsl.sh

ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  /home/linuxbrew/.linuxbrew/bin/cmake --preset cuda-release && \
  /home/linuxbrew/.linuxbrew/bin/cmake --build --preset cuda-release -j2'
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  <CudaToolEnvironment> /home/linuxbrew/.linuxbrew/bin/ctest \
  --preset cuda-release --output-on-failure --verbose'             # 8/8
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python3 scripts/benchmark_batch.py --power-mode \
  "AC/Balanced (PowerOnline=true, Charging=false, Discharging=false)" \
  --output benchmarks/results/20260719T-m4-final-72ae/batch.json'   # 2.920x PASS
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  PATH=<CUDA-Sanitizer-13.3-dir>:$PATH python3 scripts/run_cuda_memcheck.py \
  --target build/cuda-release/benchmarks/batch_benchmark \
  --output-dir benchmarks/profiles/20260719T-m4-final-72ae-memcheck -- --test'
                                                                    # 0 leak/errors
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python3 scripts/openai_smoke_test.py \
  --output benchmarks/results/20260719T-m4-final-72ae/smoke.json'   # 20/20 terminal

mise exec -- uv lock --check
mise exec -- uv run ruff format --check scripts tests
mise exec -- uv run ruff check scripts tests
mise exec -- uv run mypy scripts                                  # 19 source files
mise exec -- uv run python -m unittest discover -s tests -v       # 35/35，无 skip
xcrun clang-format --dry-run --Werror app/* include/tlie/* src/* tests/* benchmarks/*.cu
cmake --preset cpu-debug && cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure --verbose             # 6/6；604 checks
```

剩余风险：M4 是单进程、单 GPU、固定 20-slot/Batch 4 的有限实现；调度公平性只覆盖本轮短 smoke，
未做 30 分钟稳定性、OOM、复杂长短混合负载或跨引擎对照，这些仍属于 M5。`temperature`/`top_p`
已验证协议与执行路径，但当前 deterministic sampler 不代表随机采样质量。Batch 报告是单次正式
10-sample baseline，尚无跨日/跨功耗模式方差结论。

### M5：横向对照、稳定性与交付

#### 工作内容

- 使用统一 prompt、seed、采样和模型比较 PyTorch、llama.cpp、自研 FP16、自研 INT8。
- 可选比较 TensorRT-LLM；不把无法在 Ampere/当前依赖版本运行视为项目失败。
- 记录 TTFT、TPOT、总 tok/s、峰值显存、batch throughput。
- 执行 30 分钟持续负载和上下文混合负载。
- 增加 CPU ASan/UBSan、CUDA memcheck 或 Compute Sanitizer。
- 编写构建、模型准备、benchmark、已知限制和性能解释文档。

#### 退出条件

- 同一脚本能生成 CSV/JSON/Markdown 对比报告。
- 30 分钟持续负载无 crash、NaN、显存持续增长和请求泄漏。
- Compute Sanitizer 关键测试无越界或 data race 报告。
- README 不依赖作者机器上的绝对路径和隐式环境。

#### M4 复审、缺口修复与 M5 验收记录（2026-07-19 CST）

状态：M4 的全部必需退出条件经独立重跑仍成立；随后只实现并完成 M5，未实现 M6 或其他后续
范围。每个 CUDA configure/build/test/Sanitizer/benchmark 段之前均在 Mac 执行
`TLIE_WSL_DIR=tinyllama-inference-engine ./scripts/sync_to_wsl.sh`，WSL 固定镜像为
`/home/xs/tinyllama-inference-engine`，只产生 build、测试、Sanitizer 和 benchmark artifact，未在
WSL 编辑源码或运行 Git。

M4 只读复审不是复用旧绿测：重新运行服务 smoke、Batch benchmark、Compute Sanitizer memcheck、
完整 CUDA CTest，并检查 CTest inventory、skip/disabled/Not Run、0 tests、环境 gate 和 backend
fallback。复审未发现必需缺口，因此没有以 M5 改动补写 M4 结论。

| M4 退出条件/审计项 | 结论 | 本轮复审证据 |
|---|---|---|
| 流式和非流式 OpenAI-compatible smoke | verified | `benchmarks/results/20260719T-m4-reaudit/smoke.json`：models、nonstream、SSE 和错误路径通过 |
| 20 并发均完成、取消或超时且无悬挂 | verified | 12 completed、4 timeout、4 cancel；最终 queued/active/KV 均为 0，failed 为 0 |
| Batch 4 总吞吐 >= Batch 1 的 1.5 倍 | verified | 复审 Batch 1/4 比值 `2.77961x`，token 完全一致，最高 71 C，未记录 thermal slowdown |
| 取消/超时后 KV block 回收 | verified | smoke 最终 `kv_blocks_used=0`、`kv_utilization=0` |
| CUDA batch memory safety | verified | M4 memcheck 复跑：0 leak、0 errors |
| skipped、disabled、0 tests、gating、fallback、未覆盖路径 | non-finding | CUDA 8 enabled、0 disabled、8/8；kernel 249 checks；无 skip/Not Run/0-tests/fallback。M4 的长稳态和跨引擎路径确属 M5，未错误算作 M4 缺口 |

M5 增加严格的 1800 秒混合负载 runner、PyTorch 与 pinned llama.cpp 实机 adapter、统一报告
aggregator，以及明确区分 memcheck/racecheck 的 Sanitizer wrapper。比较 workload 固定 context 128、
output 32、greedy、seed 0、3 warmups、10 samples；四个 engine 必须产生完全相同的 32-token
sequence。构建、模型准备、测量边界和限制已写入 `README.md` 与
`docs/adr/0005-m5-stability-and-comparison.md`。

实现过程中发现并修复了一个真实 data race：首次 racecheck 在 softmax reduction 的共享内存复用
处报告 3 个 hazard（2 errors、1 warning）。在读取 BlockReduceMax 结果后增加 block barrier，最终
racecheck 为 0 hazards。30 分钟负载的 v1 因未排除预热窗口而不满足 settled-memory 判定，v2 因
单次 `/metrics` timeout 被错误当成 server crash，v3 在确认 20 个 HTTP worker 可造成 probe 饥饿
后主动停止；只有修正为“独立检查 child liveness、记录 probe miss、至少 10 个有效 scheduler
样本且最终 drain 成功”的 v4 被采信。失败运行均未计作通过。

正式稳定性 artifact 为 `benchmarks/results/20260719T-m5-formal-v4/stability.json`：请求持续时间
1800 秒，观察时间 `1830.840s`，并发 20；927 short、934 medium、940 long 完成，200 cancel、
160 timeout，server 累计 submitted/completed/cancelled/failed 为 `3161/2801/360/0`。最终
queued/active/KV 均为 0；丢弃前 60 个预热采样后取得 1556 个显存样本，首尾窗口中位数
`4023/4066.5 MiB`，增长 `43.5 MiB <= 64 MiB`；metrics miss 为 0。温度 49--87 C 且出现
software thermal slowdown，因此此运行只作为稳定性证据，不作为性能基线。

统一报告位于 `benchmarks/results/20260719T-m5-comparison/final-v2/`，同一命令同时生成
`report.json`、`summary.csv` 和 `report.md`。四行 exact token 均通过：

| Engine | TTFT ms | TPOT ms | 总 tok/s | 峰值 MiB | Batch 4 tok/s | thermal clean |
|---|---:|---:|---:|---:|---:|---|
| PyTorch FP16 | 28.488 | 27.408 | 36.467 | 2120.872 | 139.452 | yes |
| llama.cpp GGUF F16 | 22.684 | 6.890 | 134.944 | 3594（process-wide） | 408.259 | yes |
| TLIE FP16 | 1278.521 | 10.398 | 19.989 | 2122 | 56.394 | yes |
| TLIE hybrid INT8 | 1293.656 | 10.351 | 19.820 | 1264 | unavailable | no |

TLIE INT8 已采用单独降温后的 INT8-first 顺序，仍在 57--86 C 期间出现 `Active`，所以其 token、
显存和正确性可采信，吞吐必须带 thermal qualification。TLIE INT8 尚无 Batch 4 实现，报告为
unavailable，没有借用 FP16 数字。llama.cpp 峰值来自进程级 `nvidia-smi`，不可冒充 allocator-local。

| M5 退出条件/审计项 | 结论 | 最终证据 |
|---|---|---|
| 单一脚本生成 CSV/JSON/Markdown 跨引擎报告 | verified | `compare_engines.py` 一次生成 `final-v2/` 三种格式；PyTorch、llama.cpp、TLIE FP16/INT8 四行齐全且 32-token exact |
| 30 分钟无 crash、NaN、持续显存增长和请求泄漏 | verified | formal-v4：1800 秒 workload；failed 0；response metrics finite；settled growth 43.5 MiB；最终 queue/active/KV 为 0 |
| 关键测试无越界或 data race | verified（修复后） | final memcheck：0 leak/0 errors；final racecheck：kernel 249 checks、`0 hazards displayed (0 errors, 0 warnings)`；首次 3 hazards 已修复 |
| README 无作者绝对路径和隐式环境 | verified | README 记录 Mac/WSL 边界、固定 llama.cpp commit、模型准备、依赖、命令和限制；源码/文档扫描未发现 `/Users/<name>` 或 `/home/xs` 硬编码 |
| skipped、disabled、0 tests、environment gating、fallback | non-finding | Python 43/43、CPU 6/6、CUDA 8/8；CTest no-tests action 为 error；未见 skip/disabled/Not Run/0-tests/backend fallback。实际 PyTorch CUDA 与 llama.cpp CUDA executable 均运行，未用 stub |

实际验收命令摘要（每个下列 WSL CUDA 段前均单独执行强制同步）：

```bash
TLIE_WSL_DIR=tinyllama-inference-engine ./scripts/sync_to_wsl.sh
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  <CudaToolEnvironment> cmake --preset cuda-release && \
  cmake --build --preset cuda-release -j2 && \
  ctest --preset cuda-release --output-on-failure --verbose'       # 8/8
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python3 scripts/load_test.py --concurrency 20 --duration 1800 \
  --output benchmarks/results/20260719T-m5-formal-v4/stability.json'
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  PATH=<CUDA-Sanitizer-13.3-dir>:$PATH python3 scripts/run_cuda_memcheck.py \
  --tool memcheck --target build/cuda-release/benchmarks/batch_benchmark \
  --output-dir benchmarks/profiles/20260719T-m5-final-memcheck -- --test'
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  PATH=<CUDA-Sanitizer-13.3-dir>:$PATH python3 scripts/run_cuda_memcheck.py \
  --tool racecheck --target build/cuda-release/tests/kernel_tests \
  --output-dir benchmarks/profiles/20260719T-m5-final-racecheck'
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  .venv/bin/python scripts/benchmark_pytorch.py ... && \
  python3 scripts/benchmark_llamacpp.py ... && \
  python3 scripts/benchmark_int8.py --comparison-only ... && \
  python3 scripts/benchmark_batch.py ... && \
  python3 scripts/compare_engines.py ...'

mise exec -- uv lock --check
mise exec -- uv run ruff format --check scripts tests
mise exec -- uv run ruff check scripts tests
mise exec -- uv run mypy scripts                                  # 23 source files
mise exec -- uv run python -m unittest discover -s tests -v       # 43/43，无 skip
xcrun clang-format --dry-run --Werror app/* include/tlie/* src/* tests/* benchmarks/*.cu
cmake --preset cpu-debug && cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure --verbose             # 6/6，ASan/UBSan
```

剩余风险：本轮跨引擎对照只有单 GPU、单日、单一 context-128 workload；没有跨功耗模式方差。
INT8 performance row 受 thermal slowdown 污染，不能用作干净吞吐基线；其 Batch 4 仍未实现。
TensorRT-LLM 是可选项，本轮未运行。测试矩阵中的真实 OOM 注入仍为 **unverified**，不能由 timeout、
client disconnect 或普通显存压力替代。30 分钟负载的最长 context 为 512、单 server process，不能
外推为多进程或无限长稳定性。PyTorch peak 是 allocator-local，llama.cpp peak 是 process-wide，
两者只按来源并列展示。WSL 项目 `.venv` 使用 `uv.lock` 中精确版本和 wheel SHA 建立，但
`uv sync --offline` 因缓存 key 不匹配失败；最终使用项目级 `uv pip` 安装并实测 CUDA 可用，未进行
任何全局安装。

#### M5 再审、验收契约修复与最终恢复（2026-07-19 CST）

本轮以当前 Mac checkout 为唯一源码重新审计 M5。仓库仍为 0 tracked files，普通 `git diff` 为空
仍不能证明无改动。只读检查和修复前 43/43 Python 绿测发现四个未覆盖的验收缺口：正式 load CLI
接受任意正时长，1 秒也能标记 M5；聚合器未强制 3 warmups/10 samples 和外部报告 provenance；
memcheck wrapper 只要求 error summary、未要求计划声称的 zero-leak summary；PyTorch thermal 只看
运行前后两个点。这些均先判为 **failed**，不是沿用旧 artifact 的通过结论。

修复后新增 regression 将正式负载固定为 1800 秒/并发 20；对照报告必须校验固定 prompt、3/10、
source model pin、source before/after、in-workload GPU monitor、真实 engine 状态和 pinned llama.cpp
commit；memcheck 同时要求 `0 bytes leaked` 与 `0 errors`；PyTorch 在每个 warmup、Batch 1 和 Batch 4
测量后采集 GPU 状态。Python 测试增至 45/45。

第一次修复后正式复跑在约 4 分钟处真实失败：一个正常请求等待 150 秒后客户端 timeout，负载
exit 1。源码追踪确认 `Engine::Submit` 在持有 tokenizer mutex 时继续申请 engine mutex，而 scheduler
在持有 engine mutex 时由 `EmitToken` 申请 tokenizer mutex，形成相反锁序并可持续死锁。现把编码锁
限制在 `Tokenizer::Encode` 的最小作用域，入队前释放。失败运行未生成/未计作通过；锁修复后重新
build、smoke、Sanitizer，并完整重跑 1800 秒。

最终稳定性 artifact 为 `benchmarks/results/20260719T-m5-lockfix-formal/stability.json`，对应 88-file
可执行源码快照 `193bc776b63a24a9dbbde77e0eac74799ce92254c8e008d5f45b37dc54925ca6`，运行前后完全
一致。请求阶段 1800 秒，观察 `1830.337s`；20 并发下 967 short、974 medium、971 long 完成，
200 client cancel、160 timeout，server 累计 submitted/completed/cancelled/failed 为
`3272/2912/360/0`。1499 个有效 scheduler/GPU 样本、metrics miss 0；最终 queued/active/KV 均为
0。丢弃 60 个预热样本后，首尾 30-sample 显存中位数均为 `3954 MiB`，增长 `0.0 MiB`。最高
87 C 且出现 thermal slowdown，因此仍只作为稳定性而非性能证据。

最终跨引擎报告为 `benchmarks/results/20260719T-m5-lockfix-comparison/final/`，同一命令生成 JSON、
CSV、Markdown；四个 engine 的固定 32-token greedy sequence 完全一致：

| Engine | TTFT ms | TPOT ms | 总 tok/s | 峰值 MiB | Batch 4 tok/s | thermal clean |
|---|---:|---:|---:|---:|---:|---|
| PyTorch FP16 | 30.729 | 27.304 | 36.505 | 2120.872 | 137.438 | yes |
| llama.cpp GGUF F16 | 22.374 | 6.604 | 139.373 | 3700（process-wide） | 460.775 | yes |
| TLIE FP16 | 1237.705 | 9.555 | 20.862 | 2122 | 59.955 | yes |
| TLIE hybrid INT8 | 1317.612 | 10.289 | 19.553 | 1264 | unavailable | no |

INT8 峰值显存下降 `40.434%`，吞吐下降 `7.139%`，仍是 memory pass / speed non-finding。INT8-first
运行期间观察到 `Active`，吞吐保持 thermal-qualified；没有 Batch 4 INT8 路径。llama.cpp peak 仍是
进程级 `nvidia-smi`，不可与 allocator-local peak 直接等同。

| M5 退出条件/审计项 | 结论 | 本轮最终证据 |
|---|---|---|
| 同一脚本生成 CSV/JSON/Markdown 对比报告 | verified（修复后） | `m5-lockfix-comparison/final/` 三格式；四 engine 齐全、3/10、provenance pins 和 32-token exact 均硬校验 |
| 30 分钟无 crash、NaN、持续显存增长和请求泄漏 | verified（修复后） | 首次复跑死锁失败；修复锁序后 formal 1800 秒 exit 0、failed 0、显存增长 0 MiB、最终 queue/active/KV 为 0 |
| Compute Sanitizer 关键测试无越界或 data race | verified（修复后） | `m5-lockfix-memcheck`：0 bytes leaked/0 errors；`m5-lockfix-racecheck`：249 checks、0 hazards/errors/warnings |
| README 无作者绝对路径和隐式环境 | verified | 构建、模型、Mac/WSL、正式时长、provenance、thermal 采样和限制均显式记录；路径扫描无作者 home 硬编码 |
| skipped、disabled、0 tests、gating、fallback | non-finding | Python 45/45、CPU 6 enabled/6 passed、CUDA 8 enabled/8 passed；无 skip/Not Run/0-tests/backend fallback。一次错误 13.3 `bin` injection PATH 在首个 API 前硬失败，改用已验证的 13.3 sanitizer 目录并同步重跑，未计作通过 |

最终命令摘要；每个 WSL CUDA 段前均单独运行第一行：

```bash
TLIE_WSL_DIR=tinyllama-inference-engine ./scripts/sync_to_wsl.sh
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  cmake --build --preset cuda-release -j2 && \
  <CudaToolEnvironment> ctest --preset cuda-release --output-on-failure' # 8/8
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python3 scripts/openai_smoke_test.py ...'                            # 20/20 terminal
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  PATH=<CUDA-Sanitizer-13.3-dir>:$PATH python3 scripts/run_cuda_memcheck.py \
  --tool memcheck --target build/cuda-release/benchmarks/batch_benchmark -- --test'
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  PATH=<CUDA-Sanitizer-13.3-dir>:$PATH python3 scripts/run_cuda_memcheck.py \
  --tool racecheck --target build/cuda-release/tests/kernel_tests'
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python3 scripts/load_test.py --concurrency 20 --duration 1800 ...'   # PASS
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  .venv/bin/python scripts/benchmark_pytorch.py ...'                  # exact tokens
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python3 scripts/benchmark_llamacpp.py ...'                          # exact tokens
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python3 scripts/benchmark_int8.py --comparison-only --int8-first ...'
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python3 scripts/benchmark_batch.py ... && python3 scripts/compare_engines.py ...'

mise exec -- uv lock --check
mise exec -- uv run ruff format --check scripts tests
mise exec -- uv run ruff check scripts tests
mise exec -- uv run mypy scripts
mise exec -- uv run python -m unittest discover -s tests -v           # 45/45
xcrun clang-format --dry-run --Werror app/* include/tlie/* src/* tests/* benchmarks/*.cu
cmake --preset cpu-debug && cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure                          # 6/6
```

剩余风险：真实 OOM 注入仍为 **unverified**，它属于测试矩阵但不是 M5 四项显式退出条件；不能由
timeout 或显存采样替代。稳定性只覆盖单 GPU、单进程、context <=512 和 bounded Batch 4。INT8
吞吐仍受 thermal slowdown 污染且无 Batch 4。最终 CUDA build 出现跳过不兼容静态
`libdl/librt/libpthread.a` 的 nvlink warning，但 target 链接、CTest 和运行均成功；尚未证明在另一
Linux toolchain 上无该 warning。仓库仍无 tracked baseline/commit。`PLAN.md` 当前只定义 M1--M5，
不存在可执行的 M6 或其他下一里程碑，因此本轮在恢复并完成 M5 后停止，没有自行扩展范围。

#### M5 当前再审与同源码快照契约修复（2026-07-20 CST）

本轮再次以 Mac checkout 为唯一源码真源，从 0 tracked files、空普通/缓存 diff 的当前状态开始
复审；空 diff 仍然只是没有 Git baseline，不能证明工作树未变化。修复前 45/45 Python tests 通过，
但只读证据链发现跨引擎聚合器仅验证每个输入报告自己的 source snapshot 前后不变，没有要求
PyTorch、llama.cpp、TLIE FP16/INT8 和 Batch 4 输入来自同一个 tree SHA-256。不同源码版本的旧报告
因此可被混合后标记为 `complete: true`，使“可复现对照”退出条件判为 **failed**。

新增 regression 在修复前为 0/1，确认 mixed hashes 未被拒绝；现聚合器要求全部输入共享一个有效
64-character tree SHA-256，同时把该 hash 写入 JSON、CSV 和 Markdown。TLIE paired report、可选
INT8-first report 和 Batch 4 report 也先做内部同源检查。修复后 Python tests 为 46/46；`uv lock`、
Ruff、mypy 和 C/C++ format 通过；macOS ASan/UBSan `cpu-debug` inventory 为 6 enabled、0 disabled，
CTest 6/6 通过。使用昨日五份同源输入在 `/tmp` 重建三格式报告成功，并显式写出共同 hash
`193bc776...`；这只验证修复后的聚合逻辑，不替代本轮真实 GPU benchmark。

本轮 RTX 3080 复验尚未完成。第一次 Mac rsync 成功生成 88-file snapshot
`0dd45d041338f94b4595aef30627612acb469ff14f5ea90ba5ce67c7ffd6d31d`，但随后 `my-wsl`
对应的 `xscom.local` 无法解析，已知 `192.168.1.222:2222` 也连续超时且 ARP 为 incomplete。无输出的
SSH 命令已中止为 exit 130，未计作 configure/build/test 结果；文档修复后还必须重新同步新的 source
snapshot。未在 WSL 编辑源码、运行 Git 或安装/升级依赖。

| M5 退出条件/审计项 | 当前结论 | 本轮证据或缺口 |
|---|---|---|
| 同一脚本生成 CSV/JSON/Markdown 可复现对比报告 | failed -> repaired locally；GPU unverified | mixed-source regression 先红后绿；46/46；旧同源输入可重聚合，但当前源码的四引擎 benchmark 尚未复跑 |
| 30 分钟无 crash、NaN、持续显存增长和请求泄漏 | unverified | 目标机离线，未能重跑当前 source snapshot 的 1800 秒正式负载；不沿用昨日结果作为本轮复验 |
| Compute Sanitizer 关键测试无越界或 data race | unverified | 目标机离线，本轮 memcheck/racecheck 未运行 |
| README 无作者绝对路径和隐式环境 | verified | 当前 README/ADR 已补充跨输入同源契约；路径与隐式环境静态审计未发现新问题 |
| skipped、disabled、0 tests、gating、fallback | local non-finding / CUDA unverified | Python 46/46、CPU 6 enabled/6 passed；CUDA inventory、smoke 和实机 backend 尚未取得本轮输出 |

因此 M5 当前不能依据本轮证据重新标为全部通过，也不能进入任何下一里程碑。待目标机恢复后，
必须在每个 CUDA build/test、smoke、Sanitizer、1800 秒负载和四引擎 benchmark 前重新从 Mac 同步，
并让所有 comparison inputs 记录同一个新 source-tree SHA-256。`PLAN.md` 仍只定义 M1--M5；不存在
可实施的后续里程碑。

#### M5 当前源码缺口修复与最终复验（2026-07-20 CST）

状态：**M5 已在当前可执行源码快照上完成复验。** 本轮先完整只读审计 M5 的代码、测试、已有
artifact、CTest inventory、Git 状态和上一节未完成记录。仓库仍为 `No commits yet`、0 tracked
files，普通/缓存 diff 为空仍不能证明没有改动。只读审计没有沿用旧绿测，而是发现并修复以下
验收缺口后才开始 GPU 复验：

- 跨引擎聚合器会接受 NaN/Inf、非十六进制的 64 字符伪 SHA-256 和空 thermal sample；新增 3 个
  regression 修复前 0/3、修复后 3/3，并要求 finite positive metrics、`[0-9a-f]{64}` 与至少一个
  合法 slowdown state。
- llama.cpp adapter 的 Batch 4 原先只测一次，却与 Batch 1 共用 `samples=10` 声明；commit 也只由
  CLI 自报。现要求 server 必须位于 clean checkout 的 `build-cuda/bin`，实际 `git rev-parse HEAD`
  等于 pinned commit，并把 Batch 4 改为 3 warmups、10 samples 的中位数。
- 30 分钟 runner 原先只要求 queue/active/KV drain，没有硬校验 server terminal accounting。现要求
  `submitted = completed + cancelled + failed` 且 `failed = 0`，并把完整 accounting 写入报告。
- server 在 CUDA `ForwardBatch` 期间收到 cancel/deadline 后，返回时仍可能发 token 并把最后一步
  标成 completed；现于发 token 前再次检查取消/deadline。服务端 output tok/s 原先用首 token 后的
  `N-1` 间隔除以 `N` token，现与 TPOT 使用相同的 `N-1` 边界。增强 smoke 实跑最后一步 in-flight
  timeout 并验证 `tok/s ~= 1000 / TPOT`。
- `ResetSlot` 失败后不再把未清空 slot 放回 allocator；非法 server port 不再由未捕获 `stoi`
  abort，而是在模型加载前返回稳定 `invalid_port` JSON。
- smoke artifact 新增 source snapshot 前后验证；README 修复 llama.cpp 的 Mac/WSL cache 变量作用域、
  GGUF 单向复制命令和生成/消费路径，并移除作者机器的 CMake 绝对路径。

修复后的 Mac 非 CUDA验证：

```bash
mise exec -- uv lock --check                                      # 48 packages
mise exec -- uv run ruff format --check scripts tests             # 32 files
mise exec -- uv run ruff check scripts tests                      # 通过
mise exec -- uv run mypy scripts                                  # 23 source files
mise exec -- uv run python -m unittest discover -s tests \
  -p 'test_*.py' -v                                               # 52/52，无 skip
xcrun clang-format --dry-run --Werror app/* include/tlie/* \
  src/* tests/* benchmarks/*.cu scripts/wsl_cuda_ld_audit.c        # 通过
bash -n scripts/sync_to_wsl.sh                                    # 通过
cmake --preset cpu-debug && cmake --build --preset cpu-debug      # 通过
ctest --preset cpu-debug --output-on-failure                      # 6/6；604 checks
```

所有 CUDA configure、build、inventory、CTest、smoke、Sanitizer、benchmark 和稳定性命令前，
均从 Mac 单独执行
`TLIE_WSL_DIR=tinyllama-inference-engine ./scripts/sync_to_wsl.sh`。WSL 只构建、测试、Sanitizer 和
benchmark，未编辑或反向复制源码。最终 pre-PLAN 被测源码为 89-file snapshot
`60fea5928169ec3f65962c97dcf47d6dcfe58739aea310d1d29859924b472abc`；所有报告的 source before/
after 完全一致。本段 PLAN 更新发生在所有验证之后，不属于被测 executable snapshot。

第一次远端 configure 因非交互 PATH 没有 `cmake` 而硬失败；使用主机既有的
`/home/linuxbrew/.linuxbrew/bin/cmake` 重跑通过，没有安装或升级工具。一次非法端口 shell harness
误用了 zsh 只读变量 `status` 而失败，改名后重跑通过；两次环境/harness 失败均未计作产品通过。

```bash
# 每段前均执行上述 Mac source sync
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  /home/linuxbrew/.linuxbrew/bin/cmake --preset cuda-release'
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  /home/linuxbrew/.linuxbrew/bin/cmake --build --preset cuda-release -j2'
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  <CudaToolEnvironment> /home/linuxbrew/.linuxbrew/bin/ctest \
  --preset cuda-release --output-on-failure --verbose'             # 8/8
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python3 scripts/openai_smoke_test.py \
  --output benchmarks/results/20260720T-m5-reaudit-60fe/smoke.json'
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  PATH=<CUDA-Sanitizer-13.3-dir>:$PATH python3 scripts/run_cuda_memcheck.py \
  --tool memcheck --target build/cuda-release/benchmarks/batch_benchmark \
  --output-dir benchmarks/profiles/20260720T-m5-reaudit-60fe-memcheck -- --test'
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  PATH=<CUDA-Sanitizer-13.3-dir>:$PATH python3 scripts/run_cuda_memcheck.py \
  --tool racecheck --target build/cuda-release/tests/kernel_tests \
  --output-dir benchmarks/profiles/20260720T-m5-reaudit-60fe-racecheck'
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python3 scripts/benchmark_batch.py ... && \
  .venv/bin/python scripts/benchmark_pytorch.py ... && \
  python3 scripts/benchmark_llamacpp.py ... && \
  python3 scripts/benchmark_int8.py --comparison-only ... && \
  python3 scripts/compare_engines.py ...'
ssh my-wsl 'cd /home/xs/tinyllama-inference-engine && \
  python3 scripts/load_test.py --concurrency 20 --duration 1800 ...'
```

CUDA inventory 为 8 enabled、0 disabled，完整 CTest 8/8；CUDA kernel target 为 249 checks，batch
target 真实运行，没有 skip、`Not Run`、0-tests、disabled 或 CPU/backend fallback。Compute
Sanitizer 2026.2.1.0 的 batch memcheck 为 `0 bytes leaked`、`0 errors`；kernel racecheck 为
249 checks、`0 hazards displayed (0 errors, 0 warnings)`。

增强 smoke 的 models、非流式、SSE、temperature/top-p、错误形状、最后一步 in-flight timeout 和
20 并发均通过；并发结果为 12 completed、4 timeout、4 client-cancel。包括预备 smoke 在内，最终
server metrics 为 submitted/completed/cancelled/failed `24/15/9/0`，queue/active/KV 均为 0，
source before/after 为 `60fea592...`。

正式 Batch 1/4 报告为 `benchmarks/results/20260720T-m5-reaudit-60fe/batch.json`：context 128、
32 output tokens、3 warmups、10 samples，Batch 1/4 总吞吐为 `19.264/55.957 tok/s`，比值
`2.905x`，tokens 完全一致；59 个 workload monitor 样本为 62--86 C，software thermal slowdown
始终 `Not Active`。

统一四引擎报告位于 `benchmarks/results/20260720T-m5-reaudit-60fe/comparison/`，同一命令生成
JSON/CSV/Markdown，四行共享 `60fea592...` 并通过 32-token exact gate：

| Engine | TTFT ms | TPOT ms | 总 tok/s | 峰值 MiB | Batch 4 tok/s | thermal clean |
|---|---:|---:|---:|---:|---:|---|
| PyTorch FP16 | 29.271 | 27.749 | 35.959 | 2120.872 | 138.196 | yes |
| llama.cpp GGUF F16 | 24.564 | 7.977 | 117.734 | 3841（process-wide） | 408.803 | yes |
| TLIE FP16 | 1349.511 | 10.741 | 19.019 | 2122 | 55.957 | yes |
| TLIE hybrid INT8 | 1674.434 | 13.445 | 15.302 | 1264 | unavailable | no |

llama.cpp checkout 实测为 clean pinned commit
`571d0d540df04f25298d0e159e520d9fc62ed121`，Batch 4 明确记录 3/10。TLIE paired INT8 的峰值
显存下降 `40.434%`，吞吐下降 `20.111%`，仍是 memory pass / speed non-finding；INT8 观察到
thermal slowdown，吞吐只作 qualified 结果，且没有 Batch 4 路径。

正式稳定性报告为 `benchmarks/results/20260720T-m5-reaudit-60fe/stability.json`：请求阶段 1800 秒，
观察 `1830.289s`，并发 20；901 short、914 medium、913 long 完成，194 client-cancel、154 timeout。
server submitted/completed/cancelled/failed 为 `3076/2728/348/0`，terminal accounting 精确守恒；
1498 个 scheduler/GPU 样本、metrics miss 0，最终 queue/active/KV 为 0。丢弃 60 个预热样本后，
首尾 30-sample 显存中位数为 `4351/4152.5 MiB`，增长 `-198.5 MiB <= 64 MiB`。最高 87 C 且
出现 software thermal slowdown，因此该运行只作为稳定性证据，不作为性能基线。

| M5 退出条件/审计项 | 当前结论 | 本轮最终证据 |
|---|---|---|
| 同一脚本生成 CSV/JSON/Markdown 可复现对比报告 | verified（修复后） | 四个真实 engine、同一 source hash、3/10、finite/thermal/provenance 和 32-token exact 全部硬校验 |
| 30 分钟无 crash、NaN、持续显存增长和请求泄漏 | verified（修复后） | 1800 秒、failed 0、accounting 守恒、显存增长 -198.5 MiB、最终 queue/active/KV 为 0 |
| Compute Sanitizer 关键测试无越界或 data race | verified | memcheck 0 leak/errors；racecheck 249 checks、0 hazards/errors/warnings |
| README 无作者绝对路径和隐式环境 | verified（修复后） | Mac/WSL 变量作用域、GGUF 单向复制、cache 路径和依赖命令均显式；作者 home 扫描无发现 |
| skipped、disabled、0 tests、environment gating、fallback | non-finding | Python 52/52、CPU 6 enabled/6 passed、CUDA 8 enabled/8 passed；失败的 PATH/harness 均硬失败并重跑，未计作通过 |
| late cancel、tok/s 算术、请求计数与 artifact provenance | failed -> repaired | 增强 smoke、3 个聚合红测、3 个 runtime-contract 红测及正式 1800 秒报告共同复验 |
| 必需退出条件剩余缺口 | non-finding | M5 四项显式退出条件无 `failed` 或 `unverified` |

剩余风险：真实 OOM 和 CUDA `ResetSlot` failure injection 仍为 **unverified**；修复选择 fail-closed，
但没有可重复故障注入证明该路径。Sanitizer 仍覆盖 batch/kernel target，不直接 instrument HTTP
thread-pool 生命周期。稳定性只覆盖单 GPU、单进程、context <=512 和 bounded Batch 4；出现热节流，
不能外推为无降频性能。INT8 无 Batch 4 且本轮速度更慢。nvlink 仍警告跳过不兼容静态
`libdl/librt/libpthread.a`，虽 build/CTest/运行均成功，尚未证明另一 Linux toolchain 无该 warning。
仓库仍无 tracked baseline/commit。`PLAN.md` 只定义 M1--M5，不存在下一未完成里程碑，因此本轮在
修复并复验 M5 后停止，没有自行扩展 M6。

## 5. 总体验收标准

| 类别 | 验收标准 |
|---|---|
| Tokenizer | 固定语料 token ID 与参考实现完全一致 |
| 数值 | FP16 核心算子与 reference 在记录的容差内通过 |
| 生成 | Greedy 前 32 token 与参考实现一致 |
| 长上下文 | 128/512/2048/4096 无 NaN、越界和崩溃 |
| FP16 性能 | 固定环境下 decode 不低于约 90 tok/s 基线 |
| INT8 | 吞吐提升 >= 20% 或显存下降 >= 25% |
| Batch | Batch 4 总吞吐 >= Batch 1 的 1.5 倍 |
| 并发 | 20 并发可完成、取消和超时 |
| 服务 | OpenAI-compatible 流式/非流式 smoke test 通过 |
| 稳定 | 30 分钟持续负载无泄漏、崩溃和 NaN |
| 对照 | PyTorch、llama.cpp、自研 FP16/INT8 报告可复现 |

## 6. 计划验收命令

具体 preset 和脚本在实现时确定。CUDA 命令只允许在 3080 Laptop 环境作为最终证据。

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure

cmake --preset cuda-release
cmake --build --preset cuda-release
ctest --preset cuda-release --output-on-failure

compute-sanitizer --tool memcheck ./build/cuda-release/tests/kernel_tests
mise exec -- uv run python scripts/compare_logits.py
mise exec -- uv run python scripts/benchmark.py --contexts 128 512 2048 4096 \
  --batch 1 --power-mode "AC/high-performance"
mise exec -- uv run python scripts/profile_cuda.py --tool both

# M5/M6 实现后才启用：
python scripts/openai_smoke_test.py --base-url http://127.0.0.1:8080/v1
python scripts/load_test.py --concurrency 20 --duration 1800
```

## 7. 测试矩阵

| 层级 | 必测内容 |
|---|---|
| CPU 单元 | tokenizer、loader、shape、sampling、错误处理 |
| CUDA 单元 | RMSNorm、RoPE、Softmax、GEMM/GEMV、Attention、KV |
| 数值 | FP32/FP16/INT8 对齐、greedy token、长上下文 |
| Sanitizer | CPU ASan/UBSan、Compute Sanitizer |
| 性能 | kernel、prefill、decode、batch、context、显存 |
| 服务 | schema、流式、取消、timeout、错误映射 |
| 并发 | 混合 prompt、长短请求、公平性、KV 回收 |
| 稳定 | 30 分钟持续负载、OOM、客户端断开 |
| 对照 | PyTorch、llama.cpp、可选 TensorRT-LLM |

## 8. Benchmark 记录模板

每次正式性能结论必须记录：

- 日期与 git commit；当任务禁止 commit 且远端不复制 `.git` 时，必须改记 Mac 源码树快照、
  commit `null` 和 dirty 状态，不能伪造 commit
- 操作系统
- GPU 完整名称和显存
- NVIDIA driver 与 CUDA 版本
- 笔记本电源/功耗模式
- GPU 温度和是否降频
- 模型文件 checksum
- dtype/量化格式
- prompt tokens、output tokens、batch、并发
- warmup 次数和测量次数
- TTFT、TPOT、tok/s、峰值显存

## 9. 主要风险

- **笔记本降频污染结果**：固定电源模式、预热、记录温度并重复测量。
- **只看速度忽略错误**：所有优化必须先通过 logits/token regression。
- **INT8 无实际收益**：以 profiler 和 non-finding 作为诚实结论。
- **范围膨胀到 vLLM**：只实现 1.1B、单 GPU、有限 batching。
- **CUDA 环境漂移**：将驱动外的项目依赖固定在 `.mise.toml`、CMake preset 或容器中。
- **第三方基线不公平**：统一模型、输入、生成参数、warmup 和计时边界。

## 10. 工具安装记录

未安装任何全局软件、CUDA 工具或驱动。M1 增加可删除的项目级依赖；M2 经用户明确授权后，
只把新版 Compute Sanitizer/Nsight `.deb` 解压到 WSL 用户目录，没有调用 `apt install`、修改系统
CUDA toolkit 或替换 driver。下表的 SHA-256 均在解压前校验。

| 时间 | 工具 | 安装命令 | 原因 | 卸载命令 |
|---|---|---|---|---|
| 2026-07-13 | Python M1 工具依赖（锁定于 `uv.lock`） | `mise exec -- uv sync --all-groups` | 模型校验/转换、PyTorch golden、lint 和类型检查 | 删除项目 `.venv/` |
| 2026-07-13 | SentencePiece `31646a4`、nlohmann/json `9cca280` | `cmake --preset cpu-debug`（CMake FetchContent；固定 codeload tarball SHA-256） | C++ tokenizer 与严格 JSON 配置/测试读取；避免首次 WSL 配置下载完整 Git 历史 | 删除项目 `build/` |
| 2026-07-19 | cpp-httplib `5814e121dfb5049f72a5c3956c3c8961b40da78b` | `cmake --preset cuda-release`（CMake FetchContent；固定 commit） | M4 loopback HTTP/SSE 服务；避免依赖 WSL 全局 HTTP 库 | 删除 WSL mirror 的 `build/cuda-release/_deps/cpp_httplib-*` |
| 2026-07-19 | PyTorch `2.7.1+cu126`、Transformers `4.53.2`、tokenizers `0.21.2`（项目 `.venv`，版本锁定于 `uv.lock`） | 将 lock 中 SHA `c33360cfc2edd976c2633b3b66c769bdcbbf0e0b6550606d188431c81e7dd1fc` 的 torch wheel 单向复制到 WSL cache，再以项目级 `uv pip install` 安装精确版本 | M5 真实 PyTorch CUDA baseline；`uv sync --offline` cache-key 不匹配，未采信其失败结果且未全局安装 | 删除 WSL mirror 的 `.venv/` 和对应用户 cache wheel |
| 2026-07-19 | llama.cpp `571d0d540df04f25298d0e159e520d9fc62ed121` | Mac 用户 cache 固定 commit；WSL 用户 cache 以 `GGML_CUDA=ON` 构建 `llama-server`/`llama-quantize` | M5 真实 llama.cpp CUDA baseline；不污染项目源码或全局环境 | 删除 Mac/WSL 的 `$HOME/.cache/tlie-baselines/llama.cpp-571d0d540df04f25298d0e159e520d9fc62ed121/` |
| 2026-07-16 | CUDA Sanitizer 13.0.85（SHA `5913520009ecc86be1c62b5793b032f81fdffdfcd4493da6212e14c3dc1f35a4`） | 下载 `.deb` 后 `dpkg-deb -x` 到 `~/.local/opt/tlie-cuda-tools/cuda-sanitizer-13.0.85` | 先验证较新 WSL sanitizer；仍无法完成注入，保留诊断基线 | `rm -rf ~/.local/opt/tlie-cuda-tools/cuda-sanitizer-13.0.85` |
| 2026-07-16 | CUDA Sanitizer 13.3.75 / Compute Sanitizer 2026.2.1.0（SHA `ab5467d9473adfb528481d1b6166b9bc718ab668563a3466d0f2c5d8dd27aa4f`） | 下载 `.deb` 后 `dpkg-deb -x` 到 `~/.local/opt/tlie-cuda-tools/cuda-sanitizer-13.3.75` | 取得当前 WSL/driver 可用的真实 memcheck | `rm -rf ~/.local/opt/tlie-cuda-tools/cuda-sanitizer-13.3.75` |
| 2026-07-16 | Nsight Systems 2026.1.3.425（SHA `c7309f1c9850f66a9eb95e7215883b8e8e439df6f65ea0cecd81e6b0181a4e83`） | 下载 `.deb` 后 `dpkg-deb -x` 到 `~/.local/opt/tlie-cuda-tools/nsight-systems-2026.1.3` | 生成可导入的 CUDA kernel timeline/NVTX report | `rm -rf ~/.local/opt/tlie-cuda-tools/nsight-systems-2026.1.3` |
| 2026-07-16 | Nsight Compute package 2026.2.1.5 / CLI 2026.2.1.0（SHA `6829651ceeb0c3f65890b9f727b74d1e550fed58c454e11c2c87442295e4eb70`） | 下载 `.deb` 后 `dpkg-deb -x` 到 `~/.local/opt/tlie-cuda-tools/nsight-compute-2026.2.1` | 采集 Ampere kernel bandwidth/occupancy/launch metrics | `rm -rf ~/.local/opt/tlie-cuda-tools/nsight-compute-2026.2.1` |
