# Thread Scaling Controlled Experiment Plan（153 服务器）

## 1. 目标与结论边界

本实验直接回应 W2，并为 W3 提供重复性和方差证据。目标是在同一台服务器、同一 workload 和显式资源预算下，测量代表性 algorithm/library/system 实现随并行查询数增加的吞吐扩展。

本实验只支持以下结论：

- 不同代表实现的单线程基线、吞吐扩展和并行效率；
- 原 Figures 16--19 中的性能差异有多少可能来自并行度；
- 在固定硬件预算下，代表性 library/system 实现是否具有可量化的扩展能力。

本实验不把实现差异归因成普遍的“算法层、库层、数据库层”因果差异。原 Figures 16--19 应改称 native-configuration performance landscape；“frontier algorithms match systems”只能在新实验的定量结果支持时保留。

## 2. 已确认的固定设置

| 项目 | 设置 |
|---|---|
| 服务器 | 153：`<remote_server>` |
| 实验盘 | `<remote_data_mount>` |
| Workload | `app_reviews-384-euclidean-filter` |
| 基向量数 | 原文件为 277,936；正式运行不得截断为旧 ACORN 日志中的 270,000 |
| 维度 | 384 |
| 距离 | Euclidean / L2 |
| 查询数 | 10,000；保持原 filter workload 和 ground truth |
| Top-k | 100 |
| 设备 | CPU only |
| 内存预算 | 16 GiB，与论文中 W1--W3 的原硬件配置对齐；启动前用原 manifest 再核对一次 |
| 索引构建线程 | 固定 `B=28`；每个方法/索引参数只构建一次并持久化，查询阶段复用同一份索引 |
| 查询并行级别 | `T = {1, 7, 14, 28}` |
| 查询任务并发上限 | `T=28` 最多 4 个任务；`T=14` 最多 8 个任务；`T=7/1` 由实时 MemAvailable 和 CPU affinity 自动计算 |
| 重复规则 | 代表点独立进程运行 3 次；若 QPS 的 CV > 10% 或方法排序不稳定，则补到 5 次 |

153 与 186 同构，但本实验的全部 T 点都在 153 上完成，不拼接 186 的 T=1 结果。

## 3. 方法与过滤语义

| 层级 | 方法 | 本实验中的实现语义 |
|---|---|---|
| Algorithm | ACORN | 原生 attribute-aware graph search；过滤结构准备时间必须按统一计时边界处理 |
| Library | HNSWlib-HNSW | 当前 harness 为扩大候选集后进行 post-filter，不表述成 HNSWlib 原生 inline filtering |
| System | Milvus-HNSW | 数据库过滤路径；计入客户端、RPC、服务端检索和结果反序列化 |
| System | Weaviate-HNSW | inverted-index allow-list 与 HNSW 路径；记录实际 filter strategy 和版本 |

RangePQ、iRangeGraph 和 DiGRA 保留在 native single-thread 图中，不进入 thread-scaled 图。

## 4. 并行度和资源预算

T 表示并发查询 worker 数，同时也是该配置可使用的固定 CPU-slot 预算。153 有 112 个 online logical CPU；本实验固定使用 logical CPU affinity，使用户指定的 28×4 和 14×8 并发上限可在同一硬件上实现。SMT、NUMA 拓扑和 cpuset 会写入 manifest，不在不同方法之间切换：

| T | 并发查询 worker | CPU-slot 预算 |
|---:|---:|---:|
| 1 | 1 | 固定 cpuset 中的 1 个 logical CPU |
| 7 | 7 | 固定 cpuset 中的 7 个 logical CPU |
| 14 | 14 | 固定 cpuset 中的 14 个 logical CPU |
| 28 | 28 | 固定 cpuset 中的 28 个 logical CPU |

主结果采用 **outer-concurrency profile**，以避免把 ACORN 的单查询并行和数据库的客户端并发混在同一条曲线上：

- 每个实现的单个 query worker 内部线程固定为 1；
- T 表示同时运行的 query worker 数和整个配置的 logical CPU-slot budget；
- HNSWlib 使用 T 个 query worker 且单次 index call 内部线程固定为 1；ACORN 使用 `ACORN_SEARCH_THREADS=T`，但其原生 `IndexACORN::search` 在 query ID 上执行 `omp for`，因此 T 仍表示并发查询 worker，而不是单条 query 的内部并行；
- Milvus/Weaviate 使用 T 个 client/query workers，服务进程仍由系统管理，但受同一 cpuset 和 memory budget 限制；
- 如需展示 ACORN/HNSWlib 的 native intra-query threading，另做标注为 supplementary native-thread profile，不与主图的 outer-concurrency 曲线合并。

执行约束：

- 不动态切换 SMT/NUMA；每个任务分配不重叠的 logical CPU cpuset，记录实际 CPU topology 和 affinity；
- 查询任务按 T 分档排队：`T=28` 的有效并发上限为 4，`T=14` 的有效并发上限为 8；T=7/1 由调度器根据实时 MemAvailable、logical CPU 数和实测 RSS 自动收缩或放宽；
- 同一索引可被多个只读 query-only 进程复用，但同一个数据库 collection/service 只允许一个 build owner，查询阶段使用固定 collection 名称和持久化 volume；
- 建图阶段默认串行（每个 build task 内部 `B=28`），避免多个大索引同时构建造成内存和 IO 峰值；
- 对数据库，T 是客户端并发查询数和整个实验的 CPU 资源预算，不声称它等于数据库内部线程数；
- 索引只保留一份，不能通过 T 份索引副本获得扩展；
- 数据库容器及其依赖进程必须包含在同一 16 GiB 内存约束中；
- 所有线程环境变量和容器 CPU/memory 限制写入 manifest。

HNSWlib 的 post-filter 自适应重试必须在主 scaling profile 中关闭或固定：使用固定 `post_filter_mode=fixed` 和 `oversampling=2`，禁止查询之间共享递增的 effective `ef`。否则 T 增大时某个 worker 可能改变全局 index 的 `ef`，Recall/QPS 变化就不再只反映并行度。

## 5. Recall 控制和代表参数选择

### 5.1 选择规则

四种实现无法保证完全相同 Recall。已有结果中 Weaviate 约为 0.949，而 ACORN/HNSWlib 的上限约为 0.894/0.905。因此不强制相同 Recall 数值，采用以下可复现实验契约：

1. 将 query ID 按固定随机种子划分为 tuning 20% 和 final evaluation 80%；
2. **完整参数网格只在 T=1 上运行一次**，但所有网格索引先统一以 `B=28` 构建并保存；T=1 阶段只做 query-only 调参，不重复建图；
3. 在 tuning split 上从完整网格中选择每种方法最多两个点：一个满足 `Recall@100 >= 0.88` 的最高 QPS 点，另一个较高 Recall 点（若该方法存在稳定的高 Recall 区间）；
4. 选定后冻结索引参数和搜索参数，在 T={1,7,14,28} 及所有重复中均不改变；查询进程只加载已校验的持久化索引；
5. 论文只报告冻结后的 final evaluation split 结果。全网格的单次结果只用于调参和覆盖性审计，不作为重复统计或 headline 排序证据。

跨方法绝对 QPS 仅在同时报告实际 Recall 的前提下解释；W2 的主结果是同一方法相对 T=1 的 speedup/efficiency。

### 5.2 T=1 参数网格

参数网格沿用已有正式 reviewer grid 和 W3 repeatability 的有效 HNSW 组合，避免新实验引入一套无法与旧结果对齐的参数空间：

| 方法 | 构建参数网格 | 搜索参数网格 | T=1 点数（不含无效点） |
|---|---|---|---:|
| HNSWlib-HNSW | `(M,efC)={(32,100),(32,400),(64,100),(64,400),(128,400)}` | `efSearch={100,200,400,800,1600}` | 25 |
| Milvus-HNSW | 同上 | `ef={100,200,400,800,1600}` | 25 |
| Weaviate-HNSW | 同上；记录实际 `maxConnections/efConstruction` | `ef={100,200,400,800,1600}`；`ef=-1` 仅作为实现默认值审计，不混入主网格 | 25 |
| ACORN | 固定原 app_reviews 结构 `M=256, efC=300, M_beta=512`；`gamma={100,120}` | `efSearch={100,200,400,800,1600}` | 10 |

HNSW 三种实现的公共锚点为 `M=64, efConstruction=400`，并保留 `efSearch={100,200,400,800,1600}`；这与已有 W3 repeatability 配置一致。这里的公共锚点只统一 HNSW 家族参数，不强行把 ACORN 的 `M/gamma` 解释成 HNSW 的同义参数。

因此，预计 T=1 探索规模为 85 个有效 query 点（4 个方法）；实际只构建并保存 17 份索引（HNSWlib/Milvus/Weaviate 各 5 份，ACORN 2 份），之后复用这些索引进行 T=1 query-only 调参；最多 8 个代表点进入四档线程扩展，不会把 85 个点全部乘以 4 个线程级别和 3 次重复。

### 5.2.1 与已有实验的对齐

| 已有实验 | 已使用的参数 | 本实验如何复用 |
|---|---|---|
| 186 正式 reviewer grid | HNSW 五组 build points；`efSearch={100,200,400,800,1600}` | 作为 T=1 tuning grid，不另造一套跨度更窄的网格 |
| 186 W3 repeatability | `M=64, efConstruction=400`；3 次独立运行；2×1000 warm-up | 作为三种 HNSW 实现的公共 anchor 和重复协议 |
| 186 W4 P0-C | `M=64, efConstruction=400`、`efSearch=400`、5 轮阶段微基准 | 只复用计时字段和阶段命名，不把 W4 的阶段实验混入 scaling 主结果 |
| 153 single-thread range experiment | T=1、完整网格一次、代表点 3 次、CV>10%补5次 | 复用重复和 build-cost 规则；不把 Wide/Tight MSong workload 混进 app_reviews 主 scaling 曲线 |

因此，新增的参数覆盖是有来源的；线程扩展只施加在冻结代表点上，线程档位固定为 T={1,7,14,28}。

### 5.3 待 153 smoke 核验的代表点候选

以下参数来自已有单线程/原始曲线，只是启动候选，不是正式冻结结果：

| 方法 | 候选索引参数 | 候选搜索参数 | 已有结果用于 sanity check |
|---|---|---|---|
| ACORN | `M=256, efConstruction=300, gamma=100/120, M_beta=512` | `efSearch∈{100,200,400,800,1600}` | 旧日志 gamma=120 的 Recall 约 0.88--0.89；需用 tuning split 重新选点 |
| HNSWlib-HNSW | 公共锚点 `M=64, efConstruction=400`，另保留全部 5 个 build groups | `efSearch∈{100,200,400,800,1600}` | W3 repeatability 已使用该锚点；旧 app_reviews M=256 结果只作历史 sanity check |
| Milvus-HNSW | 公共锚点 `M=64, efConstruction=400`，另保留全部 5 个 build groups | `ef∈{100,200,400,800,1600}` | 旧结果中的 `M=4` 不再作为默认代表点，只保留为兼容性审计记录 |
| Weaviate-HNSW | 公共锚点 `M=64, efConstruction=400`，另保留全部 5 个 build groups | `ef∈{100,200,400,800,1600}`；单独记录 `ef=-1` 默认行为 | 必须确认 `ef`、filter strategy 和实际 maxConnections 生效 |

“固定 index parameters”表示每个方法内部跨 T 固定，不表示四种不同实现使用数值相同的参数。T=1 全网格的作用是覆盖参数空间；T>1 只运行冻结代表点。

本实验不再把旧 app_reviews 结果中的 `M=4` 或 `M=256` 直接当作跨层公共参数；它们只保留在历史结果审计中。若某个方法不支持某个公共 HNSW 组合，必须把该点标记为 `unsupported`，不能替换成语义不同的参数后继续排名。

### 5.4 明确不扩大的轴

- 不把 `T` 扩成 2/4/16/32：`1,7,14,28` 对齐 153 的 28-core NUMA 节点和用户指定的 4 个扩展档位；
- 不在 W2 主 scaling 中再加入 MSong Wide/Tight 或第二个数据集；低/高过滤通过率由 153 的单线程 range-aware 实验回答，混入这里会同时改变 workload 结构；
- 不把 Faiss-HNSW 强行加入主网格。Faiss 是 W4 阶段微基准的 library 对照；若 153 的 Faiss 环境修复并通过 smoke，可只加一个 `M=64, efConstruction=400` anchor 做 supplementary check，不改变主实验规模；
- 不把 `ef=-1`、不同 filter strategy 或不同服务版本当成普通搜索参数点；这些只作为 Weaviate 行为审计，并单独记录。

## 6. Query 执行协议

### 6.1 Warm-up 和 cache state

- 索引加载完成后执行 2 轮 warm-up，每轮最多 1,000 条查询；
- 使用 warm page cache，不主动 drop OS page cache；
- warm-up 不计入 QPS/latency；
- 每次独立进程重复都记录索引加载、warm-up 开始/结束和测量开始/结束时间。

### 6.2 统一计时边界

- 使用 closed-loop workers 和统一 start barrier；
- final evaluation query 按 query ID 确定性分配给 T 个 worker，不重复查询；
- `QPS = N_eval / (最后一个结果完成时间 - barrier 释放时间)`；
- ACORN 计入逐查询 bitmap/allow-list 准备；
- HNSWlib 计入 oversampling、post-filter 和不足 k 时的重试；
- Milvus/Weaviate 计入客户端序列化、RPC、服务端检索、结果返回和反序列化；
- ground-truth 读取和 Recall 计算不计入 query latency。

## 7. 重复、统计与异常处理

每个冻结配置默认运行 3 个相互独立的 query-only 进程重复。每次重复必须重新启动 benchmark client，但不重建索引；数据库 service/collection 是否重启需固定并写入 manifest，不得在方法之间变化。若服务必须重启，必须从同一持久化 volume 恢复并通过 collection/index provenance 校验。

报告：

- Recall@100；
- aggregate QPS；
- per-query p50/p95/p99 latency；
- QPS 的 mean、SD、CV 和 95% CI；
- CPU utilization、core-seconds、peak RSS；
- `S_T = QPS_T / QPS_1`；
- `E_T = QPS_T / (T × QPS_1)`。

若 CV > 10% 或跨方法排序在 3 次重复间不稳定，只对相关点补到 5 次。失败、超时和不满足 Recall 门槛的运行保留原始记录，不静默删除。

## 8. 索引构建成本

构建成本与 query scaling 分开测量：

- 只对最终冻结的每种方法代表配置独立构建 3 次；
- 构建阶段固定 `B=28` 和 16 GiB 内存预算；
- 每个 T=1 网格点只构建一次并保存索引文件（或数据库持久化 collection），记录数据导入时间、索引构建时间、总 wall-clock、CPU time、peak RSS、最终 index size、索引 SHA-256/collection provenance；
- 最终冻结代表配置另做 3 次独立 build-cost 观测，仍固定 `B=28`，但 query scaling 统一复用第一次通过校验的 saved index；不得跨不同方法共用索引；
- 构建运行不与 query scaling 并发。

## 9. 153 启动前门禁

正式大实验只能在以下项目全部通过后进入 `running` 状态：

- [ ] 153 当前 CPU、内存和 `<remote_data_mount>` 可用空间满足要求，且没有冲突任务；
- [ ] 记录 CPU 型号、NUMA 拓扑、内核、容器版本、代码 commit 和 dirty state；
- [ ] 核对原 W3/app_reviews manifest，确认 16 GiB 内存预算；
- [ ] 核对 base/query/labels/filters/neighbors 的路径、大小和 SHA-256；
- [ ] 确认四种方法使用相同的 277,936 条 base、10,000 条 query 和 ground truth；
- [ ] 确认 ACORN 不再使用 `N=270000` 截断数据；
- [ ] 确认 Weaviate 的 `ef` 和 filter strategy 实际生效；
- [ ] 确认 ACORN/HNSWlib 在 T=7/14/28 下确实共享单索引并发查询，内部单 query library thread 固定为 1；
- [ ] 每种方法先运行 100 条 query 的 T=1、T=7、T=14、T=28 smoke；
- [ ] smoke 输出 Recall、QPS、错误数、实际 worker 数、CPU affinity 和完成标记；
- [ ] tuning split 选点并生成 selected-point manifest；
- [ ] tmux session、结果目录、日志目录和显式完成标记均已创建。

建议远端目录：

```text
<remote_artifact>
  manifest/
  logs/
  results/
  indexes/
  scripts/
```

建议 tmux session：`w2_thread_scaling_153`。完成标记必须是结果校验通过后写入的 `THREAD_SCALING_COMPLETE`，不能把脚本启动或 smoke 通过当成完成。

## 10. 结果图和论文表述

主图：

1. QPS vs. T，并带 95% CI；每条线同时标注固定点的实际 Recall；
2. speedup `S_T` vs. T；
3. parallel efficiency `E_T` vs. T；
4. p95/p99 latency vs. T；
5. 代表配置的单线程构建时间、peak RSS 和 index size 表。

建议论文表述：

> We conduct an additional controlled scaling study on app_reviews using one fixed hardware profile. We build each index once with 28 build threads and reuse the persisted index for query-only measurements. For each representative implementation, index and search parameters are selected on a disjoint tuning split under a common Recall@100 threshold and then frozen. We vary the concurrent-query and physical-core budget from 1 to 7, 14, and 28, report end-to-end QPS and tail latency with confidence intervals, and normalize each implementation against its own single-worker baseline. This experiment quantifies parallelism effects without treating implementation-layer labels as a causal explanation.

## 11. 当前状态（2026-09-16 09:41 CST）

- 153 门禁与基础建图：`complete`；112 个逻辑 CPU、数据哈希和存储信息已写入 `manifest/preflight.json`，17/17 份基础索引均以 `B=28` 完成并可复用；
- T=1 tuning：Milvus 25/25、Weaviate 25/25、HNSWlib 25/25、ACORN 10/10，共 85/85 全部完成。HNSWlib 的加载容量与 11 次候选池回退边界修复已在完整 tuning 上验证通过；
- ACORN 正确性修复：旧二进制会在每个单点任务内重复 32 个 efSearch，并把最后的 `efs=10` 错配到任务结果。现已替换为 `test_acorn_controlled`，每个任务只执行命令行指定的一个 efSearch；warm-up 固定为 2×1000 query，测量为 1×10000 query；旧日志保存在 `logs/full_before_acorn_controlled_restart_20260915_1431.log`；
- ACORN smoke：`T={1,7,14,28}` 的 100-query smoke 全部通过，实际 `search_threads` 与 T 一致，Recall 保持 0.9181；
- 正式实验：`manifest/selected_points.json` 已生成，8 个代表点的 `T={1,7,14,28}`、每点 3 次 final-split scaling 已完成 96/96，未出现失败；
- 独立 build-cost：去重后的代表构建配置共 21 个完成标记（ACORN 6、HNSWlib 6、Milvus 3、Weaviate 6）。Weaviate 的 6 个构建最初因复用了 scaling collection 名称而收到 HTTP 422 `class already exists`；调度器现为每个 build-cost repeat 生成独立的 `..._buildcost_repN` collection，6 个补跑均已成功。失败日志保存在 `logs/full_before_recovery_20260916_093118.log`。Milvus 的 3 个任务已完成，但 manifest 仍是 `running` 且未写入构建计时字段，暂不把 Milvus 构建时间作为论文数值；
- 正式完成：`THREAD_SCALING_COMPLETE` 已于 09:41 生成，scaling 96/96、tuning 85/85、build-cost 21/21，失败与运行中的 task-status 均为 0；
- 自动巡检：Codex heartbeat automation `153` 已启用，后续只对完成标记、失败或需要补跑的状态发出通知。

## 12. 已完成结果摘要（2026-09-16 09:41 CST）

下表是两个代表点、每点 3 次 final-split 重复的均值；QPS 按 `T={1,7,14,28}` 报告，Recall 为该方法两个代表点的均值。原始结果仍保存在 153 的 `scaling/` 和 `build_cost/` 目录。

| 方法 | Recall 均值 | QPS(T=1) | QPS(T=7) | QPS(T=14) | QPS(T=28) |
|---|---:|---:|---:|---:|---:|
| ACORN | 0.9022 | 20.4 | 57.7 | 72.9 | 89.8 |
| HNSWlib-HNSW | 0.9005 | 1.52 | 1.00 | 0.96 | 0.93 |
| Milvus-HNSW | 0.9399 | 156.0 | 551.4 | 433.2 | 342.5 |
| Weaviate-HNSW | 0.9485 | 40.0 | 240.9 | 306.7 | 330.4 |

- Milvus 的绝对 QPS 最高，但在本配置下约在 `T=7` 达到峰值，之后因服务端/客户端端到端开销出现下降；Weaviate 从单线程到多线程扩展最明显，但 `T=1` 的部分点噪声较大；ACORN 能够扩展但绝对吞吐较低；HNSWlib 在当前固定 post-filter harness 下没有随 T 扩展，且 p95/p99 随 T 显著恶化，这应作为观测结果而不是实验失败。
- 正式重复次数为 `n=3`，已采集 SD/CV，但还没有生成 95% CI。最高 CV 包括 Milvus 方法聚合的 `T=1`（约 24.6%）、Weaviate `M=32,T=1`（约 20.1%）以及 ACORN `gamma=100,T=28`（约 13.6%）；若这些点进入论文主图，应按预定规则补到 5 次并报告 CI。
- 已得到的构建成本：ACORN 约 10.225 s（索引约 1.536--1.558 GB）；HNSWlib `M=64` 约 27.6--28.4 s（约 575 MB），`M=128` 约 122--135 s（约 717 MB）；Weaviate `M=32` 约 49--50 s、`M=64` 约 96--98 s，但当前 manifest 没有真实 collection 的磁盘大小。Milvus 构建任务虽已完成，计时字段仍需一次去重后的干净复验后才能用于论文。

## 13. HNSWlib/Milvus 配置核验与小规模复测（2026-09-18）

### 目标与范围

- 不重跑完整网格，只复测此前扩展性异常的两个高 Recall 代表点：HNSWlib `M=128, efConstruction=400, efSearch=1600`，Milvus `M=64, efConstruction=100, ef=1600`；
- 查询线程档位为 `T={1,7,28}`，每档 5 次，固定 final split、查询集、top-k=100、2×1000 warm-up；HNSWlib 的独立 query 进程最多 4 个并行，分别绑定四个互不重叠的 28-core CPU 分区，Milvus 仍串行访问共享服务；
- 复用 153 上已完成的持久化索引/collection，不覆盖正式 `scaling/` 结果；新输出根目录为 `<remote_artifact>`。

### 配置核验项

- HNSWlib：适配器 query 内部固定 `index.set_num_threads(1)`；`efSearch=1600` 在 query 阶段一次设置并冻结；外层 T 个 worker 使用 taskset CPU 集合；记录候选池初始/最终大小、总候选请求量、adaptive retries、ANN traversal、predicate evaluation 与 wasted-to-useful ratio；
- Milvus：复用 collection `ts153_milvus_hnsw_m64_e100`；每个 query 使用 `search_params={"metric_type":"L2","params":{"ef":1600}}`；记录客户端 RPC、结果解码、端到端延迟，并同步采样 Milvus 容器 CPU、内存、cpuset/quota；
- 服务器检查：服务容器当前绑定 `cpuset=0-27`、CPU quota 未限制；HNSWlib 客户端的 T-sized affinity 在四个固定槽位中分配：`0-27`、`28-55`、`56-83`、`84-111`，底层库线程固定为 1。Milvus 客户端使用一个槽位并串行执行。

### 执行状态

- 已完成两个 T=1、100-query diagnostics smoke。HNSWlib 和 Milvus 均能复用现有索引/collection 并产出完整结果；Milvus 初次 smoke 暴露出按 T 传递 `--cpuset-cpus` 会把服务重建到单核，已修复为客户端按 T 绑核、Milvus 服务固定 `0-27`，固定服务 smoke 已通过。
- 返回数量核验表明过滤选择性本身会导致少于 `k=100` 的合法结果（例如 query 27 的过滤总体只有 17 条）；Milvus smoke 对这些 query 返回了全部可用结果，HNSWlib 的少数 query 还受到近似搜索 Recall 的影响。已有正式输出中各 T/重复的返回数量序列完全一致，因此不把“少于 100”误判为线程扩展故障；正式复测仍记录 min/max/histogram 和 under-k 计数。
- 已修复诊断脚本，使其保存查询进程与服务容器的最终 CPU 绑定状态、返回数量分布及服务 cpuset 元数据；四路并行 smoke 已通过。此前串行 pilot 尚未产出完整任务结果，已停止并从同一复测根目录重新排程；正式运行中 HNSWlib 最多 4 个独立任务并行，Milvus 仍串行，共 30 个正式任务。
- 结果、命令、CPU 采样和日志均保留在上述新目录；正式结论只在所有输出文件可读、各档位返回数量序列稳定且无失败后写入本节。

### 正式复测结果（2026-09-19）

- 远端完成标记：`DIAGNOSTIC_RERUN_COMPLETE`，30/30 个任务、240,000 条 query metrics 均完整；HNSWlib 的并发槽位为四个互不重叠的 28-core 分区，Milvus 服务最终保持 `cpuset=0-27`、无 CPU quota。

| 方法 | T | QPS 均值 | QPS SD | CV | Recall | p50 / p95 / p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| HNSWlib-HNSW | 1 | 1.461 | 0.051 | 3.48% | 0.929649 | 150.97 / 3461.43 / 4174.86 |
| HNSWlib-HNSW | 7 | 0.962 | 0.035 | 3.66% | 0.929649 | 1575.56 / 37077.86 / 45532.32 |
| HNSWlib-HNSW | 28 | 0.916 | 0.017 | 1.88% | 0.929649 | 6253.07 / 152136.94 / 185750.44 |
| Milvus-HNSW | 1 | 100.195 | 1.591 | 1.59% | 0.945991 | 6.50 / 27.52 / 34.00 |
| Milvus-HNSW | 7 | 391.837 | 5.510 | 1.41% | 0.945991 | 15.12 / 35.54 / 44.37 |
| Milvus-HNSW | 28 | 342.850 | 1.647 | 0.48% | 0.945991 | 69.37 / 151.66 / 208.53 |

- HNSWlib 的 T=7/T=28 相对 T=1 仅约 0.658×/0.627×，确认后置过滤路径没有扩展收益；其候选 yield 约 5.9%、wasted-to-useful 约 1,159、adaptive retries 均值约 5.32，提供了机制层证据。
- Milvus 在 T=7 达到约 3.91× T=1，T=28 回落到约 3.42×；五次重复的 CV 均低于 4%，排序稳定。少于 100 的返回数量来自过滤后可用对象数或 HNSWlib 近似召回，跨 T/重复的返回数量序列一致。
