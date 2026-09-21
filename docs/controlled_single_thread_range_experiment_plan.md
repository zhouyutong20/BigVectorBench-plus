# 受控单线程 Range-aware 多粒度实验方案

## 1. 实验目标

本实验用于统一补充以下审稿证据：

- **Rev2.O1.2**：说明 range-aware 算法预期优于通用方法的条件，并验证至少部分 benchmark workload 满足这些条件；
- **Reviewer 1 W2**：在统一线程、硬件和内存预算下进行算法层、算法库层和数据库层的受控比较；
- **Reviewer 1 W3**：报告 warm-up、缓存状态、独立重复、方差、95% 置信区间和运行间噪声；
- **Reviewer 1 W4**：测量候选浪费、谓词检查和可获得的搜索阶段指标，避免仅凭 Recall-QPS 推断根因；
- **Reviewer 2 O3**：补充索引构建时间、内存/空间成本以及 Recall-构建成本关系。

该实验只支持“在所测代表 workload、实现和固定硬件下”的结论，不用于声称所有 range-aware 方法在所有过滤 workload 上普遍优于通用方法。

## 2. 运行位置与目录

- 执行服务器：`<remote_server>`
- 数据盘：`/dev/sdi1`，挂载点 `<remote_data_mount>`
- 建议实验根目录：`<remote_artifact>`
- 建议目录结构：

```text
<remote_artifact>
  repos/          # 算法、算法库和 benchmark 代码
  data/           # HDF5 workload 和算法格式转换文件
  indexes/        # 可复用索引；按方法/构建配置隔离
  manifests/      # 数据、代码、参数、环境和校验清单
  raw/            # 原始逐配置结果
  logs/           # stdout/stderr 和资源监控日志
  summaries/      # 聚合表、Pareto 点和统计结果
  figures/        # 论文图
```

正式复制、生成或运行前必须确认上述目标路径位于 `<remote_data_mount>`，不得将大文件写入系统盘或用户 home。

## 3. Workload 设计

### 3.1 受控变量

两个 workload 使用完全相同的：

- 训练向量和查询向量；
- 数据规模和向量维度；
- 单个有序数值过滤属性及其与向量的对应关系；
- 距离度量和 `K=100`；HDF5 中保存 200 个 filtered ground-truth 邻居用于校验和避免边界不足，主文 Recall 指标统一按前 100 个邻居计算；
- 查询数量和查询顺序；
- 硬件、线程、调参预算与方法版本。

唯一主要变化是每条查询的数值区间宽度：

| Workload | 谓词 | 目标通过率 | 作用 |
|---|---|---:|---|
| Wide-range | 单属性闭区间 `[l, r]` | 约 80% | 过滤约束较弱的对照；检查 range-aware 额外结构成本 |
| Tight-range | 单属性闭区间 `[l, r]` | 约 12% | 候选稀缺条件；检查范围剪枝、候选浪费与性能变化 |

首选基础数据为现有 `msong-1filter-80a`。Tight-range 必须基于同一 msong 向量和同一单属性值重新生成查询区间与 filtered top-100 ground truth，不能用 `tiny5m-6filter-12a` 替代，因为后者同时改变规模、维度和属性数量。

### 3.2 数据完成判据

每个 workload 只有同时满足以下条件才标记为 `READY`：

1. HDF5 文件存在且可完整读取；
2. 训练/查询向量、单属性值、查询条件和 filtered ground truth 字段齐全；
3. Wide/Tight 两个文件的向量和属性数组校验和一致；
4. 每条 ground-truth ID 唯一、合法且满足对应范围谓词；
5. 抽样或完整重算验证 top-100 距离排序；
6. 记录实际 `filter_pass_rate` 的 mean、median、p5、p95、min 和 max；
7. 训练/查询规模、维度、距离度量、生成脚本版本和随机种子写入 manifest；
8. 文件大小和 SHA-256 写入 manifest。

“文件已生成”不等于“已验证可用于正式实验”。

## 4. 评测对象

保留原有算法集合：

- 算法层：RangePQ、iRangeGraph、ACORN、DiGRA；
- 算法库层：Faiss-HNSW、HNSWlib-HNSW；
- 数据库层：Milvus-HNSW、Weaviate-HNSW。

算法层沿用各自原生实现、原生参数语义和原生构建流程。不得把不同方法名称相似的查询预算参数当成相同含义；最终通过 Recall-QPS Pareto 前沿和 matched-recall 点比较。

## 5. 统一单线程协议

构建和查询阶段全部使用单线程：

- `build_threads=1`；
- `query_workers=1`；
- OpenMP、MKL、OpenBLAS、BLIS、NumExpr 等线程数均固定为 1；
- 每个任务绑定一个独立物理 CPU 核并记录 affinity；
- 算法/库进程记录 requested/effective threads；
- 数据库使用 one client worker，并将数据库容器/服务限制在固定 one-CPU budget。

如果无法证明数据库内部全部组件严格单线程，论文表述使用：

> one client query worker under a fixed one-CPU server budget

不得写成未经验证的“database internal single-thread execution”。

## 6. 索引复用

- 算法库和数据库：Wide/Tight 两个 workload 共用同一个向量索引和同一个标量索引；只切换查询区间与 ground truth；
- 算法层：沿用原实现。若其范围结构依赖查询边界或 workload-specific metadata，则分别构建并如实记录；若不依赖，则允许复用；
- 两个 workload 的 ground truth 不复用；
- 同一索引的构建成本只报告一次，不能复制成两个独立构建观察。

每次查询前必须验证加载的索引 provenance 与训练向量、属性数组和构建参数一致。

## 7. 参数网格与重复策略

### 7.1 探索阶段

- 原有完整参数网格的每个有效配置先运行 1 次；
- 用于生成初始 Recall-QPS 曲线、检查失败配置并筛选 Pareto 点；
- 无效配置、返回不足 K、过滤错误或线程不符的结果标为 invalid，不进入排名；
- 单次探索结果不能单独支撑 headline 排序或统计可靠性结论。

### 7.2 论文代表点

- 只对进入论文图表的 Pareto/matched-recall 代表点做独立复验；
- 每个代表点先完成 3 次独立进程测量；
- 报告 mean、standard deviation、95% t-confidence interval 和 `CV=std/mean`；
- 如果 `CV > 10%` 或三次运行间排序不稳定，则补到 5 次；
- 5 次后置信区间仍明显重叠时，不写确定排序，报告为在所测协议下无法与噪声区分。

## 8. Warm-up 与缓存状态

- 默认使用 warm-cache steady-state；
- 两轮 warm-up，每轮最多 1,000 条查询，warm-up 结果丢弃；
- 不主动清空 OS page cache；
- 索引在计时前完成加载并达到 ready 状态；
- 固定查询集合和顺序，配置/方法的外层运行顺序随机化或轮转；
- 正式重复优先避免与高内存带宽构建任务并行。

所有结果记录 cache state、warm-up 数量、measured query count 和独立进程 ID。

## 9. 动态并行任务数

单个实验任务为单线程，但允许多个独立任务并行。并行数不预先固定，根据服务器剩余内存和 CPU 核决定：

```text
parallelism = min(
  available_physical_cores,
  floor((MemAvailable - safety_margin) / estimated_peak_memory_per_task),
  pending_tasks
)
```

执行约束：

- 至少保留 20% 物理内存作为安全余量；
- 数据库任务额外预留容器、服务进程和 page cache；
- 首个配置运行后用 peak RSS 更新每类任务的内存估计；
- 不允许使用 swap 支撑正式结果；
- 出现 OOM、swap、严重 page reclaim 或共享资源异常时，该次结果作废；
- 每个并发任务使用独立 CPU 核、结果目录、端口、容器项目名和索引目录。

## 10. Candidate instrumentation

Candidate instrumentation 只在 Wide/Tight 的 matched-recall 代表点开启，不对全网格插桩。

正式性能结果使用 instrumentation-off；相同参数的机制诊断使用 instrumentation-on。至少记录可获得的：

```text
candidate_examined
candidate_passed
candidate_rejected
candidate_returned
predicate_evaluations
distance_computations
adaptive_retries
out_of_range_visits
range_pruned_candidates
ann_traversal_ns
predicate_eval_ns
postfilter_ns
result_finalize_ns
```

派生指标：

```text
candidate_yield = candidate_passed / candidate_examined
wasted_to_useful = candidate_rejected / max(candidate_passed, 1)
examined_per_returned = candidate_examined / max(candidate_returned, 1)
```

不同方法的计数必须同时记录 `counter_source` 和语义。无法暴露的字段写 `unavailable`；不得用最终 top-100 返回数冒充图访问节点数或距离计算次数。

## 11. 构建成本

完整参数网格的探索运行记录一次构建成本；只有进入论文的代表 build 配置进行 3 次单线程独立重建。

记录：

```text
data_load_s
training_s
scalar_or_attribute_index_s
vector_index_build_s
flush_or_save_s
index_ready_wait_s
total_time_to_ready_s
peak_rss_bytes
serialized_index_bytes
storage_delta_bytes
index_size_source
```

算法库和数据库的两个 workload 共用索引，因此每个代表配置只做 3 次独立构建，而不是两个 workload 各 3 次。若某方法只能测总构建时间，则报告总时间和计时边界，不使用端到端差值臆测内部阶段。

## 12. 输出与判定

### 12.1 主要输出

1. Wide/Tight 单线程 Recall@100-QPS 曲线及代表点 95% CI；
2. 固定 Recall 下的最佳可达 QPS 和 range-aware/general-purpose 比值；
3. pass rate、candidate yield、wasted-to-useful 和搜索计数表；
4. 构建时间/内存/空间成本表；
5. Recall@100-构建时间图；
6. workload-condition mapping：数值范围、实际通过率、规模、属性数量和结构特征。

### 12.2 结论边界

- **Positive**：range-aware 方法在重叠 Recall 区间形成更优 Pareto 点或固定 Recall QPS 更高，且候选/访问计数与预期机制一致；
- **Recall robustness only**：range-aware 方法达到通用方法未达到的 Recall，但没有重叠 Recall 下的 QPS 比较；
- **Mixed**：优势仅在 Tight、特定 Recall 区间或特定实现出现；
- **Negative/inconclusive**：未观察到优势或差异无法与运行噪声区分；相应收窄论文结论；
- **Invalid protocol**：过滤、ground truth、线程、候选补足或索引 provenance 不满足协议；不进入论文。

## 13. 正式启动前检查表

- [x] 153 上 `<remote_data_mount>` 挂载、权限和剩余空间已验证；
- [x] 实验根目录解析后确实位于 `<remote_data_mount>`；
- [x] Wide/Tight 两个单属性 workload 均达到 `READY`；
- [x] 从 186 复制的文件均有源/目标大小和 SHA-256 校验；
- [ ] 八个方法的代码版本、依赖和构建命令已记录；
- [ ] 所有构建/查询路径的 effective threads 均为 1；
- [ ] 数据库 one-CPU budget 和独立端口/容器项目名已验证；
- [ ] 代表配置选择规则在查看正式复验结果前冻结；
- [ ] 探索、正式复验和 instrumentation 结果分别存放；
- [ ] schema 能记录重复、CI、cache、candidate 和 build-cost 字段。

## 14. 当前状态（2026-08-13，数据与协议代码已落盘）

- 153 `<remote_data_mount>`：已确认挂载自 `/dev/sdi1`，约 1.7 TiB 可用；`<remote_data_mount>/ytzhou` 已具有 `ytzhou:ytzhou` 读写权限；
- 实验根目录已创建并验证位于该文件系统：`<remote_artifact>`，包含 `repos/`、`data/`、`indexes/`、`manifests/`、`raw/`、`logs/`、`summaries/`、`figures/` 和 `packages/`；
- `msong-1filter-80a`：186 源文件与 153 现有文件均为 3,400,114,688 bytes，SHA-256 均为 `b06c7ae0f5322a0568978b91bbc7f955a286562011d10c466d98ffcdfa309624`，无需重复传输；
- Wide-range 文件已落盘至 `<remote_artifact>`。HDF5 包含 `train_vec (985185,420)`、`test_vec (10000,420)`、`train_label (985185,1)`、`test_label (10000,1,2)`、`neighbors/distances (10000,200)`；完整检查确认 10,000 条查询的 200 个 ground-truth ID 全部合法、行内唯一、满足谓词且距离非降序；实际 pass rate mean/median 为 `0.8000059/0.7999929`，p5/p95 为 `0.7998467/0.8002253`，因此标记为 `READY`；
- 同一 MSong 底库的 Tight-range 文件已生成至 `<remote_artifact>`，SHA-256 为 `931227d5aeb33eec7a12f5fd2d2f70e76e573129ebeb2ad9fdd0814f57d0e1fc`。其范围以 Wide-range 查询中心为中心，固定目标宽度为属性域的 12%；实际 pass rate mean/median 为 `0.1200374/0.1200394`，p5/p95 为 `0.1194669/0.1206221`；
- Tight-range ground truth 使用 GPU FP64 全量 flat L2、CPU FP64 距离重算和 `(distance, vector_id)` 确定性重排生成。16 查询的独立 CPU 分块穷举交叉校验中，top-200 ID 和距离逐项完全一致，最大距离误差为 0；随后对全部 10,000×200 结果进行验收，无非法 ID、重复 ID、谓词违规或距离乱序，因此标记为 `READY`；
- 对 `train_vec`、`test_vec` 和 `train_label` 分别进行全内容 SHA-256 后，Wide/Tight 三组数据集的形状、类型和哈希均逐项一致；两者只改变查询范围与相应 filtered ground truth，满足受控变量要求；
- Tight-range manifest 位于 `<remote_artifact>`，记录源/目标 SHA-256、规模、维度、设备、GT 方法和通过率分布；
- `tiny5m-6filter-12a`：是六属性且底库不同，不能替代本实验 Tight-range workload；
- `artificial-corr-12-original/shuffled`：属于相关性实验，不能替代本实验的 Wide/Tight 受控范围对照；
- 186 上 `artificial-corr-12-original.hdf5` 当前仅 1 byte，是无效占位文件；shuffled 文件存在，但与本实验无关；
- 186 最小代码迁移包已生成并复制到 153：`<remote_artifact>`，大小 229,047 bytes，SHA-256 为 `d9db8cfd134461ff0f2b87b74b153efcb2e623dd37935ba36ce2e489519f57ed`；源/目标哈希一致。该包排除了 `.git`、数据、结果、日志目录、临时目录、容器卷、HDF5、旧索引和备份文件；
- 协议代码已解压到 `<remote_artifact>`；生成和验收脚本位于其 `experiments/reviewer_2026/` 下；
- 153 已存在 RangePQ、iRangeGraph、ACORN、DiGRA 和 BigVectorBench 的本地副本，不迁移其大体积历史构建目录；后续只同步经核验的协议补丁和新增脚本；
- 153 的现有 `bigvectorbench` Conda 环境可导入 NumPy、h5py 和 PyTorch（8×V100 可用），但当前 `faiss` 导入因 `libfaiss.so` 的 `__libc_single_threaded` 符号错误失败；在正式网格前必须修复并重新验证环境，不得把未运行的配置标记为可用；
- Faiss 环境问题已隔离修复：153 为 glibc 2.31，而 `libfaiss.so` 引用了该版本不存在的 `__libc_single_threaded`。已在 `<remote_artifact>` 编译 shim（SHA-256：`37fd55e7b03d3ba1d565bbd8712447df852c108b746be7631b2bf985ba2e9858`），不替换系统 glibc 或 Conda 文件；CPU Faiss-HNSW、GPU Faiss-Flat，以及实际 BigVectorBench Faiss adapter 的单线程 build/query/filter smoke test 均通过。所有 Faiss 任务必须经 `run_faiss_compat_single_thread.sh` 启动，并记录该 shim manifest；
- 正式实验：尚未启动。

## 15. 执行计划（供审阅）

### 阶段 A：环境封存与 smoke test（先做）

1. 固定服务器 `lc153`、文件系统 `/dev/sdi1 -> <remote_data_mount>`、CPU/内存预算、GPU 设备和容器/进程版本；记录 `uname`、glibc、编译器、Python、CUDA、Faiss、HNSWlib、Milvus 和 Weaviate 版本。
2. 所有构建和查询任务统一设置 `build_threads=1`、`query_workers=1`，并固定 OpenMP/MKL/OpenBLAS/BLIS/NumExpr/Faiss 线程变量为 1。
3. Faiss 只通过 `run_faiss_compat_single_thread.sh` 启动；先完成 CPU HNSW、GPU Flat 和实际 benchmark adapter 的最小构建/查询 smoke test。shim 仅作用于 Faiss 进程，禁止修改系统 glibc 或直接覆盖 Conda 动态库。
4. 每个方法先用 1,000 条向量、100 条查询验证：数据加载、索引构建、过滤谓词、Recall 计算、QPS 计时、索引保存/加载和线程记录；失败配置标记 `invalid`，不进入排名。

### 阶段 B：受控探索网格（每个精确配置 1 次）

1. 只使用同一 MSong 底库的 Wide 80% 和 Tight 12% 两个单属性 workload；算法库/数据库两个 workload 复用同一索引，只切换查询范围和 GT。
2. 方法集合冻结为：算法层 `RangePQ/iRangeGraph/ACORN/DiGRA`，库层 `Faiss-HNSW/HNSWlib-HNSW`，数据库层 `Milvus-HNSW/Weaviate-HNSW`。
3. 使用原始实现和原始参数语义；扫描现有有效参数网格 1 次，保留完整原始日志，不在此阶段开启 candidate instrumentation。Faiss adapter 的 `BVB_THREADS/BVB_QUERY_WORKERS` 和底层线程变量均由启动器固定为 1。
4. 每个配置记录 Recall@100、QPS、延迟分位数、有效返回率、失败原因、线程/CPU affinity、峰值 RSS、索引大小和索引 provenance。
5. 并行任务数按实时 `MemAvailable` 和已测 peak RSS 动态决定，至少保留 20% 物理内存余量；单任务仍是单线程，禁止 swap/OOM 结果进入汇总。

### 阶段 C：代表点冻结与统计复验

1. 每个方法在每个 workload 选取：Recall-QPS Pareto 点、共同 Recall 区间的 matched-recall 点、以及能代表高/低过滤比的点；选择规则在查看复验结果前冻结。
2. 代表点独立进程复验 3 次，报告 mean、SD、CV 和 95% t-confidence interval；不把探索单次结果当作统计结论。
3. 若 `CV>10%` 或三次运行排序不稳定，补至 5 次；5 次后仍重叠则报告“在当前噪声下不可区分”，不强行排序。
4. 每次正式查询前执行两轮 warm-up（每轮最多 1,000 条），丢弃 warm-up 结果；默认 warm-cache，不主动清空 OS page cache，并记录 cache state、warm-up 数量、query count 和进程 ID。
5. 只有 matched-recall 代表点开启 candidate instrumentation；正式性能曲线使用 instrumentation-off，机制诊断单独存放。

### 阶段 D：W2/W4 根因分析

1. W2：在固定单线程、固定硬件/内存预算下，统一绘制算法层—库层—数据库层的对照；本方案不引入 1/8/16 线程混合曲线，所有正式 Recall-QPS 比较均为 effective query worker=1。
2. W4 candidate：记录 `candidate_examined/passed/rejected/returned`、predicate evaluations、distance computations、adaptive retries、range-pruned candidates，以及可获得的 ANN traversal/predicate/post-filter/result-finalize 时间。
3. W4 system gap：在同一查询集合上分别测 in-process library search、数据库 RPC/客户端调用、结果封装和端到端路径；无法观测的阶段写 `unavailable`，不以总差值臆测具体归因。
4. 只有当 wasted-to-useful、candidate yield 或阶段时间与 Recall-QPS 差异方向一致时，才使用“候选浪费/系统封装开销与差距一致”的措辞；否则收窄为观察性结果。

### 阶段 E：构建成本与 O3

1. 探索网格记录一次构建成本；只对进入论文的代表配置独立单线程重建 3 次。算法库/数据库两个 workload 共用同一索引，因此不重复构建两次。
2. 记录 `data_load_s`、训练/属性索引/向量索引构建、flush/save、ready wait、total time-to-ready、peak RSS、serialized index bytes、storage delta 和计时边界。
3. 构建成本与 Recall@100/QPS 通过配置 ID 连接，绘制 Recall—build-time 和 QPS—build/memory/space trade-off；缺少阶段字段时保留总时间并明确 `unavailable`，不从端到端差值推断内部阶段。

### 阶段 F：停止条件与论文纳入

- 数据/GT、线程、索引 provenance 或过滤谓词校验失败：该配置作废；
- 单次探索失败不自动重跑，先记录失败原因；只有修复环境或脚本后重新生成的配置才可进入候选集；
- 统计复验若 CV 高或排序不稳定，报告不确定性而不是挑选有利运行；
- range-aware 只有在重叠 Recall 区间具有更高 QPS/Pareto，且候选/阶段计数与机制一致时，才支持“在这些 workload 条件下优于通用方法”；
- 若优势只出现在 Tight 或某个 Recall 区间，结论明确限定到该条件；没有重叠 Recall 时只报告 Recall 可达性，不宣称 QPS 优势；
- 正式网格、代表点复验、instrumentation 和 build-cost 结果分别保存，使用统一 schema 和 manifest 汇总。

## 16. 自动调度脚本与当前 smoke 状态

内存感知调度脚本位于 `experiments/reviewer_2026/auto_experiment_orchestrator.py`，同步到 153 的 `<remote_artifact>`。脚本读取 `MemAvailable`，默认保留 20% 安全余量，并按任务 peak RSS 估算并发数；每个 worker 固定 build/query/OMP/MKL/OpenBLAS/Faiss 线程为 1、绑定独立 CPU、写独立日志和 JSON manifest，并通过 `BVB_RESULTS_DIR`/`BVB_RUNTIME_DIR` 隔离结果和索引。正式 grid 可在一个 workload runner 内派生多个独立单线程 worker；当前 153 配置为 Wide=7、Tight=7，即有效单线程 worker 并行度 14，而不是把单个worker改成多线程。

已完成的调度器 smoke：Wide/Tight 两个 Faiss adapter 任务并发启动且均返回 0，日志包含 `FAISS_ADAPTER_OK`，manifest 记录数据 SHA-256、单线程字段和返回码。当前环境的 HNSWlib import 为 `ModuleNotFoundError`，相应任务自动标记 `blocked`；因此 smoke 通过不等于 HNSWlib 或正式全参数网格已完成。`smoke` 可直接运行，`grid` 已接入 Faiss runner；`representative`/`build` 保持保护模式，待代表点和构建 wrapper 冻结后再解除。
