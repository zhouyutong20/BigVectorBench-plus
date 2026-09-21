# Artifact routing for release

## Hugging Face

Upload only the five workload data files themselves as a Hugging Face dataset repository:

- the five HDF5 files in `data/external_hdf5/`;
- including the base/query vectors, filter labels or precomputed predicate passes, neighbors and distances contained by those HDF5 files.

The conversion manifests, checksums and dataset metadata stay in GitHub; they describe the Hugging Face files but are not additional Hugging Face payloads in this release.

The final Hugging Face repository ID and URL are still pending. The current five files are the newly recovered server-side workloads; the older ag_news, cc_news, app_reviews, amazon_books, MSong, Deep1M, Tiny5M, SIFT and artificial-selectivity workload assets are not all copied into this staging directory and must be added to the Hugging Face dataset if they are needed for clean reproduction.

## GitHub

Keep the reproducibility layer and the selected results in GitHub:

- benchmark code, patches, adapters and scripts;
- workload/config manifests, protocol documents and experiment-batch metadata;
- result tables for the five new datasets, including Recall, QPS, p50/p95/p99 and status fields;
- all currently available build-time records, including build records, `index_id` associations where available and index-cost metadata;
- ACORN C1-C4 diagnostics and U1 raw predicate-cost records;
- U1 `raw_passes.jsonl`, `summary.json`, five-repeat records and patch;
- figure point tables, plotting scripts, PDFs and PNGs;
- dataset manifests, checksums and the Hugging Face download URL.

## External or internal only

- Legacy search/Recall/QPS tables and legacy diagnostics are not part of this selective public release.
- Per-query traces larger than normal GitHub file limits should go to another artifact store, with a GitHub manifest and checksum; they are not workload data for the five-file Hugging Face payload.
- Built indexes, Docker runtime volumes, compiled third-party trees and private absolute paths should not be released.
- `_remote_stage/` is an internal audit area and is not a GitHub upload source.
