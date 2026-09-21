# Selective release scope — 2026-09-21

The release has been narrowed after the initial broad audit.

## Hugging Face payload

Hugging Face receives only these five dataset files:

- `artificial-corr-12-original.hdf5`
- `artificial-corr-12-shuffled.hdf5`
- `artificial-udf-12.hdf5`
- `cc_news-original.hdf5`
- `cc_news-shuffled.hdf5`

The files are staged under `data/external_hdf5/` and are checksum-verified. The final Hugging Face repository ID and URL are still pending. Dataset manifests and checksums remain in the GitHub package.

## GitHub payload

GitHub keeps:

1. Results for the four correlation datasets (`C1-C4`) under `results/raw/correlation_campaign_20260813/` and the corresponding ACORN reuse audit.
2. Results for `artificial-udf-12` (`U1`) under `results/diagnostics/U1/`.
3. All currently available build-time records under `results/build_time/`, including the legacy workload build records. These tables may contain build-related source fields, but old workload search results are not released as separate result payloads.
4. Selected new-dataset and build-time figure point tables, PDFs, PNGs and plotting scripts.
5. Code, patches, experiment protocols, configuration metadata, checksums and release manifests.

## Explicit exclusions

The old W/S/R search tables, S1/S5 diagnostics, R1/R2 diagnostic results, W3 scaling results, Milvus W4 component results and old search-only figures were moved to `_remote_stage/excluded_by_release_scope_20260921/`. They are retained for audit but are not GitHub candidates.

The older workload data assets are also not part of the five-file Hugging Face payload. If clean reproduction of those old workload experiments is later required, they need a separate data release decision.
