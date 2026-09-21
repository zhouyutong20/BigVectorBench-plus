# ACORN C1--C4 单线程索引复用搜索实验

## Plan

- Objective: 在不重复建图的前提下，补充 ACORN 在 C1--C4 上的单线程查询结果，作为与 28 线程正式结果的执行粒度对照。
- Dataset / ground truth: `artificial-corr-12-original`、`artificial-corr-12-shuffled`、`cc_news-original`、`cc_news-shuffled`；沿用各数据集已有 filtered ground truth。
- Index reuse: 复用 186 正式全网格已生成的 ACORN 混合索引；配置为 `M=64`、`M_beta=64`、`efConstruction=400`、`gamma=12`。
- Query configuration: `efSearch={100,200,400,800,1600}`，`k=100`，2 轮 × 1,000 条查询预热，1 次 measured run；`ACORN_SEARCH_THREADS=1`、`OMP_NUM_THREADS=1`。
- Build configuration: 不建图；程序仅在找不到索引文件时才构建，运行前检查四个目标索引文件均存在且非空。

## Execution

- Server / user: 186 (`<remote_server>`) / `ytzhou`
- Existing index root: `<remote_artifact>`
- Binary: `<remote_artifact>`
- New remote run directory: `<remote_artifact>`
- Completion marker: one `-----DONE-----` per dataset and four non-empty logs.
- Smoke check: `cc_news-original`, `M=64/efC=400/gamma=12`, `efSearch=400` successfully loaded the existing index and reported `[OMP] Using 1 threads for SEARCH`; observed `Recall@100=0.9169`, `QPS=161.6`.

## Results

| Dataset | Index reused | Search thread | Status |
|---|---:|---:|---|
| artificial-corr-12-original | verified | 1 | complete |
| artificial-corr-12-shuffled | verified | 1 | complete |
| cc_news-original | verified | 1 | complete |
| cc_news-shuffled | verified | 1 | complete |

### Measured points

| Dataset | `efSearch` | Recall@100 | QPS |
|---|---:|---:|---:|
| artificial-corr-12-original | 100 | 0.8587 | 193.1 |
| artificial-corr-12-original | 200 | 0.9295 | 112.2 |
| artificial-corr-12-original | 400 | 0.9727 | 66.3 |
| artificial-corr-12-original | 800 | 0.9917 | 39.4 |
| artificial-corr-12-original | 1600 | 0.9980 | 29.7 |
| artificial-corr-12-shuffled | 100 | 0.9657 | 243.3 |
| artificial-corr-12-shuffled | 200 | 0.9888 | 140.5 |
| artificial-corr-12-shuffled | 400 | 0.9970 | 80.6 |
| artificial-corr-12-shuffled | 800 | 0.9993 | 45.1 |
| artificial-corr-12-shuffled | 1600 | 0.9999 | 27.8 |
| cc_news-original | 100 | 0.8649 | 392.6 |
| cc_news-original | 200 | 0.9010 | 217.8 |
| cc_news-original | 400 | 0.9169 | 119.9 |
| cc_news-original | 800 | 0.9238 | 62.3 |
| cc_news-original | 1600 | 0.9262 | 31.0 |
| cc_news-shuffled | 100 | 0.8301 | 351.2 |
| cc_news-shuffled | 200 | 0.8742 | 195.5 |
| cc_news-shuffled | 400 | 0.8954 | 105.3 |
| cc_news-shuffled | 800 | 0.9041 | 54.9 |
| cc_news-shuffled | 1600 | 0.9073 | 27.6 |

## Interpretation and limitations

- These are query-only single-thread measurements on indices originally built under the formal 28-thread build protocol.
- They should not be treated as independent build repeats or as a replacement for the formal 28-thread Recall--QPS grid.
- All four logs contain two `load index from` lines, `[OMP] Using 1 threads for SEARCH`, five measured `efSearch` points, `-----DONE-----`, and `RUN_RC=0`.
- Remote logs: `<remote_artifact>`; completion marker: `<remote_artifact>`.
