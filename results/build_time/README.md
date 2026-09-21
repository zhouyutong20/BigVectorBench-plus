# Build-time result tables

This directory is the complete build-time subset selected from the local result collection. It includes build records for legacy workloads as well as the new datasets when available; legacy search-only result payloads are not included in the public candidate package.

- `all_parameter_groups_build_only_20260921.csv`: parameter-group table projected to build/load/index-cost fields and source provenance.
- `supplementary_build_only_20260921.csv`: supplementary build records projected to data-load, insert and index-build components where available.
- `legacy_coverage_build_summary_20260916.csv`: workload/implementation build coverage and build-time min/max summary.

The original mixed parameter/search tables are retained under `_remote_stage/excluded_by_release_scope_20260921/results/build_time_source/`; they are not part of the public candidate package.
