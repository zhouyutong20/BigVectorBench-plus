# 受控单线程 Range-aware 网格结果审计与下一步计划

更新时间：2026-08-20 11:36（Asia/Shanghai）  
执行服务器：`<remote_server>`  
实验根目录：`<remote_artifact>`

## 1. 证据边界

本次是完整参数网格的探索阶段：每个精确配置正式查询一次，`warmup-runs=2`、每轮最多 1,000 条预热查询，`query-workers=1`、`build-threads=1`、`parallelism=1`。任务之间可以并行，但单个任务仍绑定一个 CPU 核。

因此，当前结果可以用于：

- 检查 workload、过滤谓词、索引 provenance、线程设置和实现路径；
- 生成初始 Recall@100--QPS 曲线并冻结代表点；
- 找出失败配置和需要修复的 adapter。

当前结果不能直接用于：

- 报告 run-to-run 方差、95% 置信区间或稳定的 headline 排序；
- 声称数据库层和算法库层已经完成公平的跨层结论；
- 解释候选浪费、网络/RPC 或结果封装的根因；
- 把一次构建记录当作代表配置的独立构建成本。

两个 workload 使用同一 MSong 底库、同一维度和单个数值属性；实测通过率约为 Wide `0.800006`、Tight `0.120037`。两者只改变查询范围和对应 filtered ground truth，满足受控范围对照的设计要求。

## 2. 当前运行和产物完整性

截至审计时，主调度器仍以 `--mode grid` 运行。ACORN-tight 和 Weaviate-HNSW-tight 仍在运行；ACORN-wide 已通过。

| 方法 / workload | 当前产物 | 审计状态 |
|---|---:|---|
| Faiss-HNSW / wide | 25 个结果 HDF5 | 可用于初步前沿；单次运行 |
| Faiss-HNSW / tight | 25 个结果 HDF5 | 可用于初步前沿；单次运行 |
| HNSWlib-HNSW / tight | 25 个结果 HDF5 | 可用于初步前沿；单次运行 |
| HNSWlib-HNSW / wide | 仅有 workload 文件，无结果文件 | **无效/缺失**，不能计入比较 |
| DiGRA / wide、tight | 各 6 个构建配置、每个 13 个 ef 点 | 可用于初步前沿；无独立复验 |
| iRangeGraph / wide、tight | 各 25 个搜索预算点和索引文件 | 可用于初步前沿；构建时间字段不完整 |
| ACORN / wide | 12 个完成日志，204 个搜索点 | 网格任务已通过 |
| ACORN / tight | 11 个完成日志，192 个搜索点 | **仍在运行** |
| RangePQ / wide、tight | 各 4 个构建配置、每个 10 个 L 点 | 结果存在，但未达到可匹配 Recall |
| Weaviate-HNSW / wide | 8 个结果 HDF5 | 可用于初步系统层观察 |
| Weaviate-HNSW / tight | 当前 12 个结果 HDF5，仍在运行 | **未完成** |
| Milvus-HNSW / wide、tight | 没有结果文件 | **无效/缺失**；日志显示连接 `localhost:19530` 失败，虽然调度器 manifest 错误地记录为 `passed` |

这里应以“结果文件可读且数量完整”为完成判据，而不是只看 manifest 的 `returncode=0/status=passed`。旧的非 `p7` Faiss/HNSWlib manifest 不纳入本审计。

## 3. 已有 Recall--QPS 结果的初步读法

下表取各方法所有已完成探索点中，在 Recall@100 不低于目标值时的最高 QPS。它只用于代表点筛选，不是统计结论；ACORN-tight 和 Weaviate-tight 仍是部分网格。

| 方法 | Wide：最佳 QPS @ Recall | Tight：最佳 QPS @ Recall | 备注 |
|---|---:|---:|---|
| Faiss-HNSW | `519.6 @ 0.9749` | `70.0 @ 0.9763` | 同一类通用 HNSW 在 Tight 约下降 7.4 倍 |
| HNSWlib-HNSW | `[RESULT NEEDED]` | `45.2 @ 0.9951` | Wide 结果缺失 |
| DiGRA | `519.5 @ 0.9769` | `456.4 @ 0.9901` | Tight 下仍有高 Recall 点 |
| iRangeGraph | `1060.6 @ 0.9559` | `910.4 @ 0.9693` | 两个 workload 均有较高 QPS，但需核对计时边界 |
| ACORN | `254.5 @ 0.9547` | `376.7 @ 0.9622` | Tight 网格尚未完成，不能下最终排序结论 |
| Weaviate-HNSW | `0.38 @ 0.9993` | `0.67 @ 0.9966` | 数据库服务路径；Tight 仍在跑 |
| RangePQ | 无点达到 `Recall@100 >= 0.90` | 无点达到 `Recall@100 >= 0.90` | 当前最大约为 Wide `0.14`、Tight `0.27`，不能做 matched-recall QPS 比较 |
| Milvus-HNSW | `[RESULT NEEDED]` | `[RESULT NEEDED]` | 服务连接失败，不能使用 manifest 的 passed |

### 3.1 对 Rev2.O1.2 的暂时判断

当前数据已经具备“预期有利条件”：同一底库下，Tight 的实际通过率约 12%，通用 Faiss-HNSW 在约 0.976 Recall 时的 QPS 从 Wide 的约 520 降至约 70；而 DiGRA/iRangeGraph 在 Tight 仍能达到约 456/910 QPS 的高 Recall 点。这与“候选稀缺时，范围剪枝或范围结构更可能有利”的机制假设一致。

但这还不是最终证据，因为 HNSWlib-Wide 和 Milvus 两个关键对照缺失、ACORN/Weaviate-Tight 未完成、所有点都只有一次正式运行，且尚未做候选计数和匹配 Recall 复验。Section 4.3.5 目前只能写成条件假设和初步观察，不能写成稳定的算法排序。

### 3.2 对 RangePQ 的暂时判断

RangePQ 的摘要记录 `post_filter=false`，并且所有 L 点的 Recall@100 仍很低（Wide 约 `0.11--0.14`，Tight 约 `0.24--0.27`）。这可能是当前 wrapper 的过滤/primary-dimension/候选预算语义没有对齐，也可能是该参数网格没有覆盖足够大的 L；无论哪种情况，它都不能进入当前的 matched-recall 前沿。下一步应先做过滤正确性和返回候选数审计，再决定是否扩展 L、m、nlist 或 cz。

## 4. 构建成本的已有证据

当前构建字段可作为量级检查，但不能作为最终构建成本表：每个配置只构建一次，且部分 `index_size` 字段明显无效。

| 方法 / workload | 已观察构建时间 | 当前限制 |
|---|---:|---|
| Faiss-HNSW / p7 wide | 约 `666--2514 s` | `index_size=0`，空间成本不可用 |
| Faiss-HNSW / p7 tight | 约 `670--2487 s` | `index_size=0`，空间成本不可用 |
| HNSWlib-HNSW / tight | 约 `767--2737 s` | `index_size` 字段约 1.5--4.1 MB，需用实际索引文件复核 |
| RangePQ | `build_index` 约 `163--164 s`，峰值 RSS 约 3.6--4.3 GB | 尚无 3 次独立重建 |
| iRangeGraph | 索引文件约 1.15 GB | 当前 CSV 未记录可比的构建阶段时间 |
| Weaviate-HNSW / wide | `total_build` 约 `5515 s`，其中 data load 约 `5507 s` | `index_size=40` 无法解释为可靠字节数 |
| Weaviate-HNSW / tight | `total_build` 约 `2109--2211 s`，其中 data load 约 `2101--2203 s` | 仍在运行，空间字段为负值 |
| ACORN / DiGRA | `[RESULT NEEDED]` | 原生日志没有可直接对齐的构建阶段字段 |

这些数字说明数据库端到端准备成本可能占很大比例，但不能把 `total_build - index_build` 直接命名为网络、解析或封装开销。

## 5. 对审稿意见的覆盖状态

| 审稿意见 | 当前状态 | 原因 |
|---|---|---|
| Rev2.O1.2 | **部分覆盖** | Wide/Tight 条件已建立，初步趋势支持假设；缺少完整对照和复验 |
| W2 | **部分覆盖** | 线程和硬件协议已统一；Milvus/HNSWlib-Wide 缺失，且数据库/库层尚未完成可比结果 |
| W3 | **未覆盖** | 有 warm-up 记录，但没有独立 run-level 方差、CV 和 CI |
| W4 | **未覆盖** | 当前网格未开启 candidate instrumentation，也没有 RPC/解析/封装微基准 |
| 构建成本 / O3 | **部分覆盖** | 已有一次性构建时间；空间字段、阶段拆分和独立 3 次重建仍缺失 |

## 6. 下一步实验计划

### P0：先修复“结果不可用”问题

1. **完成当前任务并做产物验收**：等待 ACORN-tight、Weaviate-tight 结束；检查每个结果文件可读、配置数量完整、Recall/过滤 ID 合法、线程字段为 1。
2. **修复 Milvus 连接和完成判据**：先独立验证容器健康、`localhost:19530`、collection/端口和单 CPU budget；修复后重新跑 Wide/Tight 的完整网格。调度器增加“无结果文件或 worker 异常时不得标记 passed”的检查。
3. **补跑 HNSWlib-Wide**：查明 manifest 通过但结果目录为空的原因；补跑完整网格，并确认结果文件数与参数定义一致。
4. **RangePQ 正确性审计**：固定 `primary_dim=0`，逐查询验证返回 ID 满足范围谓词，并与同一 filtered top-100 ground truth 对齐；记录 `post_filter`、候选数、最终返回数和有效 Recall。只有达到至少一个高 Recall 点后，才继续扩大 L/m/nlist/cz 网格。

### P1：冻结代表点并获得统计证据

5. **冻结选择规则后再选点**：每个方法×workload 选择一个 Pareto 点、一个共同 Recall 区间的 matched-recall 点（优先约 0.95 或 0.99），以及必要时的高 Recall 点。若没有 Recall 重叠，只报告可达性，不强行比较 QPS。
6. **代表点独立复验**：每点先运行 3 次独立进程；报告 mean、SD、CV 和 95% t-confidence interval。若 CV>10% 或三次运行排序不稳定，再补到 5 次；5 次后仍重叠则写“在当前噪声下不可区分”。
7. **保持正式性能与机制插桩分离**：正式性能用 instrumentation-off；只在 matched-recall 代表点开启 candidate instrumentation，记录 candidate examined/passed/rejected/returned、predicate evaluations、distance computations、retries 和可测分阶段时间，并注明 `counter_source`。
8. **系统层微基准**：对同一代表配置测 in-process library、数据库 client/RPC、结果封装和端到端路径；无法观测的阶段写 `unavailable`，不从总差值臆测根因。

### P1：代表配置构建成本

9. 每个进入论文的代表 build 配置做 3 次独立单线程重建；库层/数据库层的 Wide/Tight 共用索引，不重复构建两次。
10. 记录 `data_load_s`、训练/属性索引/向量索引、flush/save、ready wait、total time-to-ready、peak RSS、实际索引文件字节数和 storage delta。若只能获得总时间，明确计时边界并保留 `unavailable`。

### P2：机制与结构分析

11. 在两个 workload 上计算 `eligible_neighbor_ratio`、局部过滤纯度、cross-filter edge ratio、过滤集合连通性等结构指标。
12. 将这些结构指标与 candidate yield、wasted-to-useful、fixed-Recall QPS 和构建成本连接起来；只有方向一致时，才把“候选浪费/结构剪枝”写成机制解释，否则保留为 observational。
13. 最后生成单线程 Recall-QPS 图、matched-recall 表、CV/CI 表、candidate 表和 Recall-构建成本图，并据此决定 Section 4.3.5 是 positive、mixed、recall-robustness-only 还是 inconclusive。

## 7. 停止和纳入规则

- 过滤谓词、ground truth、线程、索引 provenance 或产物完整性失败：配置标记 invalid，不进入排名。
- RangePQ 在修复前不做重复实验，不用低 Recall 点声称高 QPS 优势。
- 代表点复验必须使用 run-level 独立进程；同一进程内的 query round 不当作独立重复。
- 没有 Recall 重叠时，不报告 speedup；只报告 Recall 可达性或明确“无法比较”。
- 所有 headline 排序都必须等待 3/5 次复验和不确定性检查；单次网格仅作为探索证据。

## 8. 2026-08-20 定向补跑执行记录

补跑脚本：`/tmp/rerun_reviewer_20260820.sh`（服务器 153）。结果根目录：`<remote_artifact>`。

| 会话 | 任务 | 并行度 | 关键设置 | 当前状态 |
|---|---|---:|---|---|
| `rerun_hnswlib_wide_0820b` | HNSWlib-Wide | 7 个独立单线程 worker | `DOCKER_HOST` 指向不存在的 socket，使 BVB 回退到进程 RSS，避免数据库容器重启造成 stale Docker stats | 已进入运行阶段 |
| `rerun_rangepq_0820` | RangePQ Wide/Tight 诊断网格 | 2 个独立单线程 worker | `M={4,8,16,32}`，`L={32000,64000,96000,128000,160000}`，`RANGEPQ_THREADS=1` | 已进入首批配置查询 |
| `rerun_milvus_0820b` | Milvus Wide→Tight | 服务任务并行度 1 | 先轮询 `http://127.0.0.1:9091/healthz`，再运行每个 workload；Wide/Tight 串行复用同一服务 | Wide 已通过健康检查并开始运行 |

第一次启动失败的两个会话是脚本工作目录遗漏导致 `logging.conf` 找不到，已修复后重新启动；该失败不计入算法结果。短时健康检查确认：RangePQ 两个 worker 均在计算，Milvus 三个容器 healthy，HNSWlib-Wide 进程存活且未再出现 Docker `NotFound`。

## 9. 2026-08-24 进度核验（153 服务器）

远端核验时间：`2026-08-24 14:44:45 +08:00`。当前没有 `rerun_*` tmux 会话，也没有对应运行进程；本轮定向补跑已经结束。

| 任务 | 完成产物 | 结束状态 | 备注 |
|---|---:|---|---|
| HNSWlib-Wide | 30 个参数结果 HDF5（另有 1 个数据文件） | 正常结束，`2026-08-20 17:16:20` | 日志含 `[end]`，未见异常 |
| Milvus-Wide | 30 个参数结果 HDF5 | 正常结束 | 日志 `result_hdf5=30` |
| Milvus-Tight | 30 个参数结果 HDF5 | 正常结束，`2026-08-21 06:21:23` | 日志 `result_hdf5=30` 并含 `[end]` |
| RangePQ | 3 个 summary JSON | **部分失败** | 完成 `m=4` 的 3 个 nlist/cz 组合；进入 `m=8` 时因 `D=420 not divisible by m=8` 停止 |

RangePQ 的 3 个已完成组合仍只有约 `Recall@100=0.139`（日志中 `L=160000`），尚未形成 matched-recall 点。因此不能把该批结果纳入高 Recall 前沿；后续应改用 420 的合法因子作为 `m`，并先做返回 ID/过滤正确性检查。

原始大网格的 14 个任务已写入 manifest，其中 ACORN、DiGRA、iRangeGraph、Milvus、Weaviate 和 p7 版 Faiss/HNSWlib 均为 `passed`；两个旧的非 p7 Faiss manifest 仍显示 `running`，但没有活跃进程，应视为陈旧状态而不是当前运行任务。调度汇总曾记录 `failures=1`，与 Weaviate-Tight 日志在最后一个配置处中断的历史记录一致，需在最终汇总时按结果文件完整性重新判定，不能只依据 manifest。

## 10. 2026-09-02 中断实验恢复

恢复前重新检查 153：`<remote_data_mount>` 可用约 1.4 TiB，`MemAvailable` 约 490 GiB，无目标实验进程。确认两项未完成工作：RangePQ 由于非法 `m` 停止；Weaviate 的 manifest 虽为 `passed`，但实际仅有 Wide `8/144`、Tight `16/144` 个查询参数结果。

修复与验证：

- RangePQ 改用 420 的合法因子 `m={4,5,6,7}`，保留 `L={32000,64000,96000,128000,160000}`，设置 4 个独立单线程 worker，并启用 cross-batch `skip_done`；
- Weaviate adapter 在删除旧 collection 遇到连接中断时使用新 client 重连重试；原文件备份为远端 `module.py.pre_resume_20260902`；
- Weaviate 独立 smoke collection 的连接、创建和删除测试通过，标记为 `WEAVIATE_RESUME_SMOKE_OK`；
- RangePQ `m=5`、100 queries、`L=32000` smoke 通过并生成非空 JSON，标记为 `RANGEPQ_RESUME_SMOKE_OK`。

长期任务于 `2026-09-02 11:41:22 +08:00` 启动：

| tmux 会话 | 任务 | 有效并行度 | 结果/日志 | 完成判据 |
|---|---|---:|---|---|
| `resume_rangepq_0902` | RangePQ Wide/Tight 合法参数续跑 | 4 个独立单线程 worker，CPU 14--17 | 结果根目录 `<remote_artifact>`；日志 `<remote_artifact>` | 累计 32 个 summary JSON 且出现 `[end]` |
| `resume_weaviate_0902` | Weaviate Wide 后 Tight 断点续跑 | one client worker，容器 `cpuset=105` | 原 grid 结果目录；日志 `<remote_artifact>` | Wide/Tight 各 144 个结果并出现 `[end]` |

`2026-09-02 11:43:43 +08:00` 健康检查：两个 tmux 会话均存在；RangePQ 4 个 worker 各占用约一个 CPU 核并已进入训练；Weaviate 父/子进程存活，服务 ready HTTP=200、`cpuset=105`、未 OOM，正在清理旧 collection 后准备重建。系统仍有约 475 GiB `MemAvailable`。Weaviate 当前定义包含 18 个 build 配置、每个 8 个查询参数，因此该任务可能持续很多天；脚本按结果计数续跑，连续三轮无进展才停止。

## 11. 2026-09-03 进度核验

核验时间：`2026-09-03 14:12:54 +08:00`。

- RangePQ 已于 `2026-09-02 15:47:58` 正常结束，累计 32/32 个 summary JSON：Wide/Tight 各 16 个，`m={4,5,6,7}` 每个 `m` 各 4 个 nlist/cz 组合。29 个新 JSON 与此前 3 个 JSON 均通过解析检查；`resume_rangepq_0902` 会话已正常消失。
- Weaviate 仍在 `resume_weaviate_0902` 中运行。Wide 从 8/144 增长到 10/144，最新完成文件为 `M=96, efConstruction=200, ef=120`，HDF5 可读并含 10,000×100 的 neighbors/distances、10,000 个 recalls/times；当前正在同一 build 配置的第 3 个查询参数 `ef=150`。Tight 仍为此前的 16/144，需等待 Wide 完成后开始。
- 恢复日志未出现新的 `RemoteProtocolError`、Traceback、OOM、Killed 或维度不整除错误。Weaviate 容器 `cpuset=105`、ready HTTP=200、OOM=false、restart count=0。
- 服务器约有 479 GiB `MemAvailable`，`<remote_data_mount>` 仍有约 1.4 TiB 可用；当前资源正常。

Weaviate 单个查询参数包含 10,000 次单线程端到端请求，当前推进速度仍较慢。其完成判据保持为 Wide/Tight 各 144 个可读 HDF5 并在恢复日志出现 `[end]`，不能仅依据 tmux 存活或旧 manifest 的 `passed` 状态判断完成。

## 12. 2026-09-04 进度核验

远端核验时间：`2026-09-04 15:09:55 +08:00`。

- `resume_weaviate_0902` 仍在运行；Wide 已从 10 增至 13/144，Tight 仍为 16/144。最近新增 Wide 结果依次为 `ef=150/200/500` 等查询参数，文件时间持续更新；当前 Weaviate 容器 CPU 约 100%，`cpuset=105`、ready HTTP=200、OOM=false、重启次数为 0，属于正常计算状态而非空闲挂死。
- RangePQ 仍为 32/32，已正常结束；全量 summary 的最高 `Recall@100` 约为 `0.3476`，因此尚未产生约 0.95 的 matched-recall 点。
- 服务器仍有约 478 GiB `MemAvailable` 和 1.4 TiB `<remote_data_mount>` 可用空间。
- 代表点重复、W3 方差/CI、W4 candidate instrumentation 与系统微基准、代表配置 3 次构建成本、汇总表和图仍未启动；`summaries/`、`figures/` 尚无正式输出。

## 13. 2026-09-04 Weaviate 并行恢复

串行恢复会话 `resume_weaviate_0902` 已在保留已有 HDF5 的前提下停止；停止前 Wide/Tight 分别为 13/144、16/144。旧的默认 Compose 容器也已停止，未删除数据、结果或卷。

长任务改为 `/tmp/launch_weaviate_parallel_20260904.sh wide`，tmux 会话为 `resume_weaviate_parallel_wide_0904`。脚本按剩余内存动态计算并行度：保留 20% `MemAvailable`，按每个 Weaviate 实例 16 GiB 峰值估计，并受 8 个参数分片和 CPU 列表限制。本次核验得到 `MemAvailable=517137896 KiB`、8 个 CPU 槽位，因此有效并行度为 8。

每个 worker 仍是单线程配置：`query-workers=1`、`build-threads=1`、`parallelism=1`、`runs=1`、两轮 warm-up（每轮最多 1,000 条查询）、`independent-build-runs=1`，不使用 `--force`。8 个 worker 使用 CPU 104--111、HTTP 端口 18080--18087、gRPC 端口 19080--19087、独立 Compose 项目/collection/工作目录；数据和结果根目录共享但构建参数分片不重叠。原 18 个 `(M,efConstruction)` 点与 8 个查询 `ef={100,120,150,200,500,600,800,-1}` 恰好覆盖一次。

隔离 smoke 已通过（`WEAVIATE_PARALLEL_SMOKE_OK`）。启动后八个容器均 ready，worker 日志均已进入 `Start loading data with 985185 data...`，没有出现端口冲突、连接重置、OOM 或参数解析错误。Wide 仍在构建/导入阶段，完成判据保持 144 个可读 HDF5；Wide 完成后再用同一脚本启动 Tight。

资源解释：命令行仍保留 `--memory 8G` 作为 runner 的检查参数，但本次 `--local` 路径由 Compose 启动服务，`docker inspect` 显示各容器 `HostConfig.Memory=0`（无限制），与此前串行恢复使用的实际配置一致。因此 16 GiB 只用于并行度的保守峰值估计，不冒充容器硬内存上限；正式跨层比较需把该事实写入硬件/内存预算表。

## 14. 2026-09-07 进度核验

远端核验时间：`2026-09-07 11:06:34 +08:00`。

- Wide 当前 `72/144` 个 HDF5，Tight 仍为 `16/144`；`resume_weaviate_parallel_wide_0904` 仍存在，Tight 尚未启动。
- Wide 的 8 个分片中 5 个仍有运行中的 Weaviate 容器（CPU 约 100%，内存约 5.4--5.8 GiB/容器），3 个 worker 已退出；因此当前是部分完成、部分运行状态，不是全网格完成。
- w3、w5 日志出现 Docker stats 查询 `404 No such container` 的 traceback，原因是 `BaseANN.get_memory_usage()` 仍访问已被移除的旧 Docker 容器；尽管外层 worker 日志显示 `rc=0`，该异常说明对应分片不能仅依据返回码视为完整成功，需按缺失文件补跑并修复错误传播/内存统计路径。
- 当前未发现 OOM、Killed 或磁盘空间异常；`MemAvailable` 约 448 GiB，`<remote_data_mount>` 可用约 1.4 TiB。已有结果仍只作为探索性单次运行，不能直接用于论文 headline 排序。

## 15. 2026-09-08 最新结果核验

远端最后一次成功核验时间：`2026-09-08 11:16:41 +08:00`。之后的只读汇总调用受到当前工具用量上限限制，以下数字以该次核验为准。

- Wide：`81/144` 个 HDF5，9 个同配置 CSV；tmux `resume_weaviate_parallel_wide_0904` 仍存在，w0、w1、w2 仍有运行中的 Python/Weaviate 实例。
- Tight：`16/144` 个 HDF5、2 个 CSV，尚未启动新的 Tight 并行任务。
- Wide 结果中至少存在可读完整 HDF5，例如 `M=48, efConstruction=200, ef=120`：`Recall_mean=0.98308`、`QPS=0.40311`、`total_build_s=2220.86`、`p50=2.60084 s`、`wall_time=24806.91 s`。这是单次探索点，不是重复统计结果。
- w3、w5、w6 的日志含 `docker.errors.NotFound: .../containers/.../stats`；该异常发生在 `BaseANN.get_memory_usage()` 统计已移除容器时，不能把这些分片的 `rc=0` 当作完整成功。当前结果必须继续按每个 `(M,efConstruction,ef)` 文件逐点验收。

## 16. 2026-09-08 Weaviate 容器统计竞态修复与断点重启

旧会话 `resume_weaviate_parallel_wide_0904` 已停止，停止时 Wide 保留 `83/144` 个 HDF5；未删除已有结果、数据或 Docker volume。根因和修复如下：

- `BaseANN.get_memory_usage()` 会枚举并缓存宿主机上的所有容器，多个并行 worker 清理各自容器后，另一个 worker 可能继续对已删除容器调用 `stats()`，从而触发 `docker.errors.NotFound`。Weaviate adapter 现仅统计当前 `BVB_WEAVIATE_PROJECT_NAME` 下 `service=weaviate` 的活跃容器；若容器在采样期间消失则跳过，并保留进程 RSS 回退。
- `create_workers_and_execute()` 原先只 `join()` 子进程，不检查 `exitcode`，因此子进程 traceback 仍可能让父进程和 launcher 得到 `rc=0`。现在汇总所有非零子进程退出码并抛出 `RuntimeError`，避免把残缺分片误标为成功。
- 修改前远端文件分别备份为 `bigvectorbench/algorithms/weaviate/module.py.pre_memfix_20260908` 和 `bigvectorbench/main.py.pre_workerprop_20260908`。修改后 SHA-256 分别为 `ea75890a25402a8c8aa7944115ac22d6a28425491fd7c886cac733e2fd5dfdf1` 和 `d3e5423fda938b8115a55a7e4404d8fc4bd5cbf736687d7c07750eb702720b4f`。

两项定向 smoke 均通过：隔离 Compose 项目的连接、reset 和项目级内存采样输出 `WEAVIATE_PROJECT_MEMORY_KIB=36692.0`、`WEAVIATE_PARALLEL_SMOKE_OK`；故意令 benchmark 子进程失败时，父进程能够捕获非零退出码并输出 `WORKER_FAILURE_PROPAGATION_SMOKE_OK`。这只证明故障路径和隔离逻辑有效，不属于正式性能结果。

Wide 于 `2026-09-08 14:38:12 +08:00` 使用 `/tmp/launch_weaviate_parallel_20260904.sh wide` 断点重启，tmux 会话为 `resume_weaviate_memfix_wide_0908`，launcher 日志为 `<remote_artifact>`。脚本未使用 `--force`，w4、w7 因分片完整而自动跳过；仅启动 w0、w1、w2、w3、w5、w6 六个缺失分片，分别固定到 CPU 104、105、106、107、109、110，对应六个隔离 Weaviate 实例。

`2026-09-08 14:41--14:42 +08:00` 启动后核验：六个 benchmark 父/子进程均存在，六个容器各使用约 94%--99% 单核 CPU，容器内存约 0.5--1.0 GiB，正在首个配置的导入/建索引阶段；尚未新增结果，Wide 仍为 `83/144`。未发现新的 `No such container`、`RemoteProtocolError`、traceback、OOM 或 Killed；服务器约有 468 GiB `MemAvailable`，`<remote_data_mount>` 可用约 1.4 TiB。完成判据仍为 Wide 144 个可读 HDF5 且 launcher 正常结束；随后才能启动 Tight，当前不能把 tmux 存活或 smoke 通过写成实验完成。

## 17. 2026-09-08 Weaviate ETA 估算

`2026-09-08 15:01:39 +08:00` 的只读核验结果：Wide 总计 `83/144`；按固定参数分片为 w0/w1/w2 各 `10/24`、w3/w5 各 `8/16`、w6 `13/16`，w4=`16/16`、w7=`8/8`。当前 6 个未完成分片均有 benchmark 子进程和 Weaviate 容器运行，结果文件尚未因新会话增加，符合首个缺失配置仍在导入/建索引或查询阶段的表现。

对已有可读 HDF5 的实际字段统计得到：Wide 83 点的 `wall_time` 中位数约 `25,270 s`（约 7.0 h，范围约 6.7--7.8 h），Tight 已有 16 点的中位数约 `15,385 s`（约 4.3 h，范围约 4.2--4.4 h）。因此按当前固定 worker 分片、单线程、每点 10,000 查询的保守估计：

- Wide 剩余最慢分片约 14 个点，预计还需约 4--5 天，预计在 **2026-09-12 至 2026-09-13** 完成；这是 Wide 的 ETA，不是全部 Weaviate 工作的完成时间。
- Wide 完成后若立即以同一脚本启动 Tight，Tight 最慢分片约 24 个点，预计再需约 4--5 天，全部 Weaviate（Wide+Tight）预计在 **2026-09-17 至 2026-09-18** 完成。

上述是基于已观测 wall time 的运行估计，不是结果承诺；容器异常、单点耗时偏离、网络/磁盘抖动或需要补跑时应顺延。当前 Tight 尚未由本轮脚本自动启动，Wide 完成后需要先检查 `[launcher-end] status=complete` 和 `144/144`，再启动 Tight。

## 18. 2026-09-08 Tight 部分并行启动

在不影响 Wide 的前提下，`2026-09-08 15:09:06 +08:00` 增开两个 Tight 分片：

- tmux：`resume_weaviate_partial_tight_0908`；脚本：`/tmp/launch_weaviate_tight_partial_20260908.sh`；日志：`<remote_artifact>`；
- Tight w0 使用 CPU 108、HTTP/gRPC `18180/19180`，独立项目 `bvb_wv_tight_w0_p0908`；w1 使用 CPU 111、`18181/19181`，独立项目 `bvb_wv_tight_w1_p0908`；
- 两个 worker 均保持 `query-workers=1`、`build-threads=1`、`parallelism=1`、`runs=1`、两轮 warm-up，并复用已有 Tight HDF5（未使用 `--force`）。

启动后检查显示两个 benchmark 进程和容器均存活，已连接并 reset 独立 collection，进入 `Start loading data with 985185 data...`；没有 traceback、`No such container`、`RemoteProtocolError`、OOM 或 Killed。此时总有效并行度为 **8**（Wide 6 + Tight 2），内存可用约 444 GiB，Tight 结果仍为 `16/144`，处于首个配置的数据导入阶段。

## 19. 2026-09-08 Tight 扩展到全 8 分片

由于服务器仍有充足内存和空闲 CPU，继续启动 Tight w2--w7：

- tmux：`resume_weaviate_partial_tight_extra_0908`；日志：`<remote_artifact>`；实际 launch 行为 w2/w3/w4/w5/w6/w7，分别绑定 CPU 96/97/98/99/100/101，HTTP 端口 18182--18187，gRPC 端口 19182--19187；
- w0/w1 继续运行于 `resume_weaviate_partial_tight_0908`，CPU 108/111、端口 18180/18181。端口和 Compose project 均与 Wide 隔离；所有 Tight worker 仍为单线程 benchmark。

`2026-09-08 15:15:07 +08:00` 健康检查：Wide 6 个、Tight 8 个容器全部存活，Tight 容器 CPU 约 94%--100%，未出现 traceback、`No such container`、`RemoteProtocolError`、OOM 或 Killed；可用内存约 415 GiB，系统负载约 15.5。Tight 结果仍为 `16/144`，新 worker 正在首个配置的数据加载阶段，尚未到首个 HDF5 落盘点。extra launcher 的 `[grid] selected_workers=0,1` 是固定提示文本，实际 worker 以同一日志中的六条 `[launch] worker=2...7` 为准。

## 20. 并行 Tight 后的 ETA 修订

Section 17 中“Wide 完成后再启动 Tight、全部预计 2026-09-17 至 2026-09-18”的估计已被本节并行启动覆盖。当前 Wide 与 Tight 同时执行，按已观测单点 wall time 和各自最慢分片估计，Wide 和 Tight 都可能在 **2026-09-12 至 2026-09-14** 间完成；具体仍取决于首个配置建索引、查询耗时和是否需要补跑。该区间是运行 ETA，不是论文结果完成或统计复验完成时间。

## 21. 2026-09-09 153 实时进度核验

`2026-09-09 10:20:33--10:20:51 +08:00` 的只读核验显示，Weaviate 的 Wide/Tight 两组均仍在持续计算，没有 launcher 结束标记，因此不能视为完成：

- **Wide：93/144 个可读 HDF5**，较前一记录的 83/144 增加 10 个。分片进度为 w0=`12/24`、w1=`11/24`、w2=`11/24`、w3=`10/16`、w4=`16/16`、w5=`10/16`、w6=`15/16`、w7=`8/8`；w0、w1、w2、w3、w5、w6 仍有 benchmark 和容器进程运行。
- **Tight：39/144 个可读 HDF5**，较启动时的 16/144 增加 23 个。分片进度为 w0=`3/24`、w1=`3/24`、w2=`3/24`、w3=`3/16`、w4=`2/16`、w5=`11/16`、w6=`11/16`、w7=`3/8`；8 个 Tight worker 均仍在运行。
- **有效并行度：14 个独立单线程 worker**（Wide 6 + Tight 8），每个 benchmark 均保持 `query-workers=1`、`build-threads=1`、`parallelism=1`；CPU 绑定没有合并到同一核。
- 服务器 112 个逻辑 CPU 中仅使用上述 14 个计算核；核验时 `MemAvailable≈366 GiB`、swap 仅约 228 MiB、load average 约 `18.66/18.19/18.14`。未发现 `Traceback`、`No such container`、`RemoteProtocolError`、OOM 或 `Killed`。

本轮无法刷新 186 的低成本重复实验：从当前工作站连接 `<remote_server>:22` 返回 `Permission denied`。因此 186 只能沿用本地保存的 `2026-09-08 10:40:16` 快照（27/74 complete、4 running、43 pending、0 failed、0 个独立构建结果），不把该快照当作 2026-09-09 的实时完成数。

## 22. 2026-09-09 15:36 复核尝试

再次对 `<remote_server>` 和 `<remote_server>` 发起只读 SSH 核验时，两个目标均返回 `ssh: connect to host ... port 22: Permission denied`，因此本次没有新增可验证的结果文件计数、进程状态或日志状态。后续进度报告继续以 Section 21 的最后一次成功核验为准，直到 SSH 权限恢复后重新检查 HDF5、`.done`/launcher-end 标记和 worker 进程。

## 23. 2026-09-10 153/186 SSH 恢复后的实时核验

`2026-09-10 09:41--09:42 +08:00` 通过只读 SSH 重新核验成功：

- **153 / Weaviate Wide：107/144**。分片为 w0=`14/24`、w1=`14/24`、w2=`14/24`、w3=`13/16`、w4=`16/16`、w5=`12/16`、w6=`16/16`、w7=`8/8`；当前仍有 w0、w1、w2、w3、w5 五个 Wide worker 运行。
- **153 / Weaviate Tight：70/144**。分片为 w0=`7/24`、w1=`7/24`、w2=`7/24`、w3=`7/16`、w4=`6/16`、w5=`14/16`、w6=`15/16`、w7=`7/8`；8 个 Tight worker 仍在运行。
- 两组累计 **177/288** 个可读 HDF5，尚余 111 个；当前分配的 14 个单线程 worker 槽位中有 13 个仍在计算（Wide 5 + Tight 8）。两组检查均未发现 traceback、容器竞态、RemoteProtocolError、OOM 或 Killed。
- 153 资源正常：112 个逻辑 CPU，`MemAvailable≈378 GiB`，swap 使用约 2.1 GiB，load average 约 `17.92/17.98/17.99`。Wide/Tight launcher 和 tmux 会话均存在，尚无完成标记。

**186 / 低成本重复实验**的 campaign 状态为 **68/74 complete、1 running、5 pending、0 failed、0 stopped**。当前运行任务是 `query__artificial-corr-12-shuffled__weaviate-hnsw`（5 次 headline repeats，cpuset `0-27`）；Weaviate 容器约 `185.94%` CPU、4.1 GiB 内存，日志已进入第 2 个 query group 的第 3/5 次运行，未见异常。独立构建结果已落盘 **15/18** 个，剩余 3 个仍未完成；scheduler/tmux 仍存活，不能把 68/74 写成最终完成。

## 24. 2026-09-10 17:27 153 最新进度

`2026-09-10 17:27:27--17:27:32 +08:00` 的只读核验显示 153 上的两组 Weaviate 仍在持续落盘：

- **Wide：112/144**，分片为 w0=`15/24`、w1=`15/24`、w2=`15/24`、w3=`14/16`、w4=`16/16`、w5=`13/16`、w6=`16/16`、w7=`8/8`；w0、w1、w2、w3、w5 五个 worker 仍运行。
- **Tight：81/144**，分片为 w0=`8/24`、w1=`9/24`、w2=`8/24`、w3=`8/16`、w4=`8/16`、w5=`16/16`、w6=`16/16`、w7=`8/8`；w0--w4 五个 worker 仍运行。
- 合计 **193/288**，尚余 95 个 HDF5；当前实际计算并行度为 **10 个单线程 worker**（Wide 5 + Tight 5），已完成的分片不会继续占用计算核。
- 服务器仍健康：`MemAvailable≈394 GiB`、swap 使用约 1.7 GiB、load average 约 `14.39/14.48/14.55`。最近 Wide/Tight 结果分别在 16:49 和 17:15 生成；检查未发现 OOM、Traceback、RemoteProtocolError、容器竞态或 Killed，尚无 launcher 完成标记。

## 25. 2026-09-11 153 状态与 W2 证据边界

`2026-09-11 11:16:37--11:16:40 +08:00` 的实时核验得到 Wide=`121/144`、Tight=`95/144`，累计=`216/288`，尚余 72 个 HDF5；当前仍有 Wide 4 个、Tight 5 个单线程 worker 运行。服务器硬件基线为两路 Intel Xeon Gold 6330（56 物理核/112 逻辑核，2 NUMA 节点）、503 GiB RAM、8×Tesla V100 16 GiB，`<remote_data_mount>` 可用约 1.4 TiB。

对 W2 的证据边界：153 已固定主机、数据/查询/filtered ground truth、K=100、单线程和数据库 one-client worker，并按 Recall@100 的 Pareto/matched-recall 点组织跨层比较；但当前仍是探索网格（每点一次），Weaviate 尚未完成，代表点 3/5 次复验和线程扩展曲线也尚未完成。另需注意 Weaviate 的 `--memory 8G` 在 `--local` Compose 路径下没有形成实际 Docker memory cap（`HostConfig.Memory=0`），因此不能把当前运行描述成“所有层具有相同硬内存上限”。最终 W2 rebuttal 应采用固定单线程主图、补充最小 thread-scaling appendix，或明确收窄跨层结论。

## 26. 2026-09-14 153/186 实时进度核验

`2026-09-14 14:07:18--14:14:48 +08:00` 通过只读 SSH 重新核验。153 上的 Weaviate 主网格已进入最后一个 Tight 点：

- **Weaviate-Wide：144/144** 个可读 HDF5；8 个分片分别为 `24/24,24/24,24/24,16/16,16/16,16/16,16/16,8/8`。launcher 已写入 `[launcher-end] 2026-09-14T04:38:59+08:00 status=complete`。
- **Weaviate-Tight：143/144**；分片为 `24/24,23/24,24/24,16/16,16/16,16/16,16/16,8/8`，仅 w1 剩 1 个结果。w2--w7 的 launcher 在 `2026-09-14 12:30:54` 以 `status=incomplete` 结束（总计 143/144）；w1 仍由 `resume_weaviate_partial_tight_0908` 运行，尚无 `[launcher-end]`。
- 14:14:48 的进程核验显示 w1 的 Weaviate 容器仍约 `99.5%` CPU、约 `17.55 GiB` 内存，benchmark 子进程存活，属于慢查询/落盘中的正常运行状态，不是空闲退出。服务器 `MemAvailable≈474 GiB`、load average `2.04/2.11/2.14`、swap 使用约 `389 MiB`；当前主实验有效并行度为 **1 个单线程 worker**（另有与本实验无关的 Stage-E monitor）。launcher 错误筛查未发现 `Traceback`、`RemoteProtocolError`、`No such container`、OOM 或 `Killed`。
- 因而 153 Weaviate 一次性网格累计 **287/288**，仅差 Tight w1 的 1 个 HDF5。按结果文件计数，ACORN `12+12`、DiGRA `6+6`、RangePQ `32` 个 summary JSON、HNSWlib-Wide `30`、Milvus Wide/Tight 各 `30`、Faiss p7 Wide/Tight 各 `25`、HNSWlib-Tight p7 `25` 均已有产物；RangePQ 仍没有达到 matched-recall 的点，不能据此下高 Recall 前沿结论。

`2026-09-14 14:12:24 +08:00` 对 186 的状态文件复核显示，低成本 W3 campaign 已明确结束：`complete=74, failed=0, pending=0, running=0, stopped=0`；`campaign_state.json` 与 `scheduler.log` 最后更新时间为 `2026-09-11 19:08:20`，当前无 campaign tmux/进程。

当前仍未看到 153 的代表点 3/5 次统计复验、candidate instrumentation、代表配置独立构建成本和 thread-scaling appendix 产物；这些属于主网格完成后的后续阶段，不能把 `287/288` 写成整套审稿补实验已完成。
