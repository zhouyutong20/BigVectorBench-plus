# Correlation campaign interim summary

> This snapshot includes only job cells with a `.done` marker. It is descriptive: each exact build configuration currently has one independent build, so no build-to-build confidence interval or significance claim is made.

## Data quality

- Completed job cells: 60
- Completed cells with analyzable Recall/QPS points: 58
- Fully covered expected parameter grids: 42
- Total result points: 1219
- HDF5/native points: 917/302
- Valid schema-v4 Recall/QPS points: 1219
- Invalid/unreadable points: 0/0
- Points with candidate diagnostics: 161
- Points with a positive index-size measurement: 1004

## Best QPS subject to Recall >= 0.95

| Dataset | Algorithm | QPS | Selected recall | Grid points | Grid status |
|---|---|---:|---:|---:|---|
| artificial-corr-12-original | acorn | 1918 | 0.9718 | 25/25 | complete |
| artificial-corr-12-original | irangegraph | 1007 | 0.9659 | 25/25 | complete |
| artificial-corr-12-original | digra | 905 | 0.9602 | 25/25 | complete |
| artificial-corr-12-original | milvus-hnsw | 105.8 | 0.9784 | 25/25 | complete |
| artificial-corr-12-original | rangepq | 52.83 | 0.9804 | 1/1 | complete |
| artificial-corr-12-original | elasticsearch-hnsw | 45.02 | 1 | 5/25 | partial |
| artificial-corr-12-original | hnswlib-hnsw | 5.195 | 0.9936 | 25/25 | complete |
| artificial-corr-12-original | qdrant | 4.726 | 1 | 5/25 | partial |
| artificial-corr-12-original | redis-hnsw | 3.909 | 1 | 8/25 | partial |
| artificial-corr-12-original | faiss-hnsw |  |  | 25/25 | complete |
| artificial-corr-12-original | milvus-gpu-cagra |  |  | 30/30 | complete |
| artificial-corr-12-original | milvus-gpu-ivfpq |  |  | 36/36 | complete |
| artificial-corr-12-original | milvus-ivfpq |  |  | 36/36 | complete |
| artificial-corr-12-original | vearch-hnsw |  |  | 5/25 | partial |
| artificial-corr-12-original | vearch-ivfpq |  |  | 4/36 | partial |
| artificial-corr-12-shuffled | acorn | 6058 | 0.9607 | 25/25 | complete |
| artificial-corr-12-shuffled | irangegraph | 2392 | 0.9554 | 25/25 | complete |
| artificial-corr-12-shuffled | digra | 2026 | 0.9598 | 25/25 | complete |
| artificial-corr-12-shuffled | milvus-gpu-cagra | 192.5 | 0.9858 | 30/30 | complete |
| artificial-corr-12-shuffled | milvus-hnsw | 173.2 | 0.9933 | 25/25 | complete |
| artificial-corr-12-shuffled | hnswlib-hnsw | 68.13 | 0.9954 | 25/25 | complete |
| artificial-corr-12-shuffled | faiss-hnsw | 67.03 | 0.9721 | 25/25 | complete |
| artificial-corr-12-shuffled | rangepq | 51.07 | 0.9984 | 1/1 | complete |
| artificial-corr-12-shuffled | elasticsearch-hnsw | 45.95 | 0.9911 | 25/25 | complete |
| artificial-corr-12-shuffled | vearch-hnsw | 11.74 | 0.9638 | 5/25 | partial |
| artificial-corr-12-shuffled | qdrant | 4.798 | 1 | 5/25 | partial |
| artificial-corr-12-shuffled | redis-hnsw | 3.847 | 1 | 2/25 | partial |
| artificial-corr-12-shuffled | milvus-gpu-ivfpq |  |  | 36/36 | complete |
| artificial-corr-12-shuffled | milvus-ivfpq |  |  | 36/36 | complete |
| artificial-corr-12-shuffled | vearch-ivfpq |  |  | 4/36 | partial |
| cc_news-original | irangegraph | 3031 | 0.9729 | 25/25 | complete |
| cc_news-original | acorn | 2439 | 0.9589 | 25/25 | complete |
| cc_news-original | digra | 1816 | 0.9769 | 25/25 | complete |
| cc_news-original | milvus-hnsw | 175.7 | 0.9865 | 10/25 | partial |
| cc_news-original | redis-hnsw | 85.7 | 0.9924 | 25/25 | complete |
| cc_news-original | elasticsearch-hnsw | 36.91 | 0.9866 | 25/25 | complete |
| cc_news-original | qdrant | 9.715 | 0.9938 | 25/25 | complete |
| cc_news-original | hnswlib-hnsw | 3.258 | 0.9793 | 8/25 | partial |
| cc_news-original | faiss-hnsw |  |  | 25/25 | complete |
| cc_news-original | milvus-gpu-cagra |  |  | 10/30 | partial |
| cc_news-original | milvus-gpu-ivfpq |  |  | 36/36 | complete |
| cc_news-original | milvus-ivfpq |  |  | 28/36 | partial |
| cc_news-original | vearch-hnsw |  |  | 25/25 | complete |
| cc_news-original | vearch-ivfpq |  |  | 36/36 | complete |
| cc_news-shuffled | irangegraph | 2620 | 0.9682 | 25/25 | complete |
| cc_news-shuffled | acorn | 1988 | 0.9535 | 25/25 | complete |
| cc_news-shuffled | digra | 1438 | 0.9839 | 25/25 | complete |
| cc_news-shuffled | milvus-hnsw | 188.9 | 0.9981 | 25/25 | complete |
| cc_news-shuffled | elasticsearch-hnsw | 37.31 | 0.9972 | 25/25 | complete |
| cc_news-shuffled | qdrant | 9.514 | 0.9991 | 25/25 | complete |
| cc_news-shuffled | redis-hnsw | 5.315 | 0.9983 | 1/25 | partial |
| cc_news-shuffled | hnswlib-hnsw | 3.017 | 0.9824 | 3/25 | partial |
| cc_news-shuffled | faiss-hnsw |  |  | 25/25 | complete |
| cc_news-shuffled | milvus-gpu-cagra |  |  | 5/30 | partial |
| cc_news-shuffled | milvus-gpu-ivfpq |  |  | 36/36 | complete |
| cc_news-shuffled | milvus-ivfpq |  |  | 36/36 | complete |
| cc_news-shuffled | vearch-hnsw |  |  | 25/25 | complete |
| cc_news-shuffled | vearch-ivfpq |  |  | 36/36 | complete |

## Build cost by job (median across unique build configurations)

| Dataset | Algorithm | Build configs | Build median (s) | Insert median (s) | Index median (s) | Index size median (MB) |
|---|---|---:|---:|---:|---:|---:|
| artificial-corr-12-original | acorn | 5 | 377.2 |  | 377.2 | 1634 |
| artificial-corr-12-original | digra | 5 | 1.168e+04 |  | 1.168e+04 |  |
| artificial-corr-12-original | elasticsearch-hnsw | 1 | 550.9 | 538.9 | 12.05 |  |
| artificial-corr-12-original | faiss-hnsw | 5 | 124.5 | 9.042 | 117.5 | 1467 |
| artificial-corr-12-original | hnswlib-hnsw | 5 | 117.6 | 9.043 | 110.6 | 1471 |
| artificial-corr-12-original | irangegraph | 5 | 2077 |  | 2077 | 1915 |
| artificial-corr-12-original | milvus-gpu-cagra | 6 | 314 | 85.98 | 228.2 | 212 |
| artificial-corr-12-original | milvus-gpu-ivfpq | 9 | 100.9 | 81.1 | 19.77 | 351.4 |
| artificial-corr-12-original | milvus-hnsw | 5 | 194.8 | 80.7 | 114.1 | 1756 |
| artificial-corr-12-original | milvus-ivfpq | 9 | 101.4 | 79.79 | 21.87 | 394 |
| artificial-corr-12-original | qdrant | 1 | 354.4 | 344.4 | 10.05 | 1198 |
| artificial-corr-12-original | rangepq | 1 | 113.3 | 29.52 | 83.78 | 986.9 |
| artificial-corr-12-original | redis-hnsw | 8 | 1972 | 1962 | 10.05 | 0.2656 |
| artificial-corr-12-original | vearch-hnsw | 1 | 554.5 | 541.4 | 13.05 | 1.883 |
| artificial-corr-12-original | vearch-ivfpq | 1 | 458.3 | 445.3 | 13.05 | 0.9141 |
| artificial-corr-12-shuffled | acorn | 5 | 367.7 |  | 367.7 | 1634 |
| artificial-corr-12-shuffled | digra | 5 | 1.147e+04 |  | 1.147e+04 |  |
| artificial-corr-12-shuffled | elasticsearch-hnsw | 5 | 678.6 | 672.6 | 6.025 | 0.01172 |
| artificial-corr-12-shuffled | faiss-hnsw | 5 | 130 | 11.05 | 119 | 1467 |
| artificial-corr-12-shuffled | hnswlib-hnsw | 5 | 113.3 | 5.028 | 108.3 | 1471 |
| artificial-corr-12-shuffled | irangegraph | 5 | 2020 |  | 2020 | 1889 |
| artificial-corr-12-shuffled | milvus-gpu-cagra | 6 | 344.8 | 85.76 | 258.4 | 936.4 |
| artificial-corr-12-shuffled | milvus-gpu-ivfpq | 9 | 111.3 | 87.1 | 24.18 | 770 |
| artificial-corr-12-shuffled | milvus-hnsw | 5 | 162.5 | 85.89 | 76.59 | 1636 |
| artificial-corr-12-shuffled | milvus-ivfpq | 9 | 114.2 | 86.2 | 28.22 | 374.6 |
| artificial-corr-12-shuffled | qdrant | 1 | 379.6 | 367.6 | 12.06 | 1246 |
| artificial-corr-12-shuffled | rangepq | 1 | 113.7 | 30.44 | 83.2 | 986.9 |
| artificial-corr-12-shuffled | redis-hnsw | 2 | 1229 | 1225 | 4.018 | 0.5117 |
| artificial-corr-12-shuffled | vearch-hnsw | 1 | 464.2 | 451.2 | 13.05 | 0.6758 |
| artificial-corr-12-shuffled | vearch-ivfpq | 1 | 449.4 | 436.3 | 13.06 | 2.066 |
| cc_news-original | acorn | 5 | 332 |  | 332 | 1326 |
| cc_news-original | digra | 5 | 6341 |  | 6341 |  |
| cc_news-original | elasticsearch-hnsw | 5 | 574.9 | 562.9 | 12.05 | 362.3 |
| cc_news-original | faiss-hnsw | 5 | 100.3 | 11.05 | 89.22 | 1222 |
| cc_news-original | hnswlib-hnsw | 2 | 52.88 | 5.027 | 47.85 | 1148 |
| cc_news-original | irangegraph | 5 | 1379 |  | 1379 | 1037 |
| cc_news-original | milvus-gpu-cagra | 2 | 149.2 | 75.22 | 74.02 | 631.5 |
| cc_news-original | milvus-gpu-ivfpq | 9 | 99.43 | 76.23 | 23.39 | 321.7 |
| cc_news-original | milvus-hnsw | 2 | 244.4 | 73.48 | 170.9 | 2220 |
| cc_news-original | milvus-ivfpq | 7 | 102.6 | 75.63 | 26.94 | 381.5 |
| cc_news-original | qdrant | 5 | 317.6 | 307.6 | 10.04 | 1124 |
| cc_news-original | redis-hnsw | 25 | 931.6 | 923.6 | 8.031 | 0.4199 |
| cc_news-original | vearch-hnsw | 5 | 406.8 | 393.7 | 13.05 | 1.02 |
| cc_news-original | vearch-ivfpq | 9 | 399.8 | 386.7 | 13.05 | 0.7188 |
| cc_news-shuffled | acorn | 5 | 236.7 |  | 236.7 | 1326 |
| cc_news-shuffled | digra | 5 | 7435 |  | 7435 |  |
| cc_news-shuffled | elasticsearch-hnsw | 5 | 569.1 | 557 | 12.05 | 371.2 |
| cc_news-shuffled | faiss-hnsw | 5 | 96.26 | 11.05 | 85.21 | 1222 |
| cc_news-shuffled | hnswlib-hnsw | 1 | 28.6 | 7.036 | 21.56 | 1073 |
| cc_news-shuffled | irangegraph | 5 | 1464 |  | 1464 | 1073 |
| cc_news-shuffled | milvus-gpu-cagra | 1 | 168.7 | 74.43 | 94.27 | 389.8 |
| cc_news-shuffled | milvus-gpu-ivfpq | 9 | 100.5 | 76.85 | 23.47 | 355.4 |
| cc_news-shuffled | milvus-hnsw | 5 | 151.6 | 76.83 | 65.36 | 1525 |
| cc_news-shuffled | milvus-ivfpq | 9 | 104.6 | 76.93 | 27.77 | 361.2 |
| cc_news-shuffled | qdrant | 5 | 341.2 | 331.2 | 12.05 | 1127 |
| cc_news-shuffled | redis-hnsw | 1 | 1243 | 1230 | 12.05 | 0.3516 |
| cc_news-shuffled | vearch-hnsw | 5 | 404.5 | 391.5 | 13.05 | 2.293 |
| cc_news-shuffled | vearch-ivfpq | 9 | 397.7 | 384.6 | 13.05 | 1.941 |

## Original versus shuffled at Recall >= 0.95

| Dataset family | Algorithm | Original QPS | Shuffled QPS | Original / shuffled |
|---|---|---:|---:|---:|
| artificial-corr-12 | acorn | 1918 | 6058 | 0.3166 |
| artificial-corr-12 | digra | 905 | 2026 | 0.4468 |
| artificial-corr-12 | elasticsearch-hnsw | 45.02 | 45.95 | 0.9797 |
| artificial-corr-12 | faiss-hnsw |  | 67.03 |  |
| artificial-corr-12 | hnswlib-hnsw | 5.195 | 68.13 | 0.07625 |
| artificial-corr-12 | irangegraph | 1007 | 2392 | 0.4208 |
| artificial-corr-12 | milvus-gpu-cagra |  | 192.5 |  |
| artificial-corr-12 | milvus-gpu-ivfpq |  |  |  |
| artificial-corr-12 | milvus-hnsw | 105.8 | 173.2 | 0.6108 |
| artificial-corr-12 | milvus-ivfpq |  |  |  |
| artificial-corr-12 | qdrant | 4.726 | 4.798 | 0.9851 |
| artificial-corr-12 | rangepq | 52.83 | 51.07 | 1.034 |
| artificial-corr-12 | redis-hnsw | 3.909 | 3.847 | 1.016 |
| artificial-corr-12 | vearch-hnsw |  | 11.74 |  |
| artificial-corr-12 | vearch-ivfpq |  |  |  |
| cc_news | acorn | 2439 | 1988 | 1.227 |
| cc_news | digra | 1816 | 1438 | 1.263 |
| cc_news | elasticsearch-hnsw | 36.91 | 37.31 | 0.9892 |
| cc_news | faiss-hnsw |  |  |  |
| cc_news | hnswlib-hnsw | 3.258 | 3.017 | 1.08 |
| cc_news | irangegraph | 3031 | 2620 | 1.157 |
| cc_news | milvus-gpu-cagra |  |  |  |
| cc_news | milvus-gpu-ivfpq |  |  |  |
| cc_news | milvus-hnsw | 175.7 | 188.9 | 0.9305 |
| cc_news | milvus-ivfpq |  |  |  |
| cc_news | qdrant | 9.715 | 9.514 | 1.021 |
| cc_news | redis-hnsw | 85.7 | 5.315 | 16.12 |
| cc_news | vearch-hnsw |  |  |  |
| cc_news | vearch-ivfpq |  |  |  |
