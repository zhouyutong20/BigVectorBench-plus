# Experimental protocol and evidence rules

## Identity fields

Every public result row should be traceable through:

`workload_id`, `implementation`, `config_id`, `run_id`, and `experiment_batch_id`.

Build records additionally carry an `index_id`. Multiple search points using the same built index must reference the same `index_id` and must not be counted as independent builds.

## Result states

Use the following machine-readable states:

- `complete`: required build/search evidence is present and readable;
- `partial`: some expected points or fields are missing;
- `unsupported`: the implementation does not support the configuration;
- `failed`: the run was attempted and failed; preserve the reason;
- `missing`: no matching run was found;
- `smoke_only`: validation output, not a formal result;
- `pending`: not yet verified for publication.

Never encode a failed or unsupported point as zero QPS, and never silently remove it from a coverage table.

## Search statistics

For formal repeated results, preserve Recall@k, returned count, aggregate QPS, mean latency, p50/p95/p99 latency, run count, standard deviation, CV and 95% confidence interval when available. State whether p95/p99 is averaged across per-run percentiles or computed after merging query-level observations.

## Split and tuning rules

Record the query IDs and seed for tuning and final evaluation splits. Parameter selection belongs to tuning only; headline curves use the independent frozen evaluation split. A selected point must be recorded in a machine-readable manifest.

## Build and diagnostics

Record data-load time, explicit index-build time, total build time, ready/checkpoint state, index size and artifact provenance. Keep library proxy counters separate from native graph counters. Instrumented and uninstrumented performance runs must be stored separately.

## Status boundary

The staging package is an evidence organization step. It does not assert that every workload/implementation cell is complete. The coverage manifest and source notes remain authoritative for missing, unsupported, failed and pending cells.
