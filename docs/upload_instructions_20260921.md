# Upload instructions — 2026-09-21

The staging directory is not itself a Git repository. Upload the five HDF5 files to Hugging Face, then copy only the selected GitHub paths into the actual GitHub clone.

## Hugging Face

Install and authenticate the CLI:

```powershell
py -m pip install --upgrade huggingface_hub
hf auth login
```

Create a dataset repository once:

```powershell
hf repos create YOUR_HF_USER/bigvectorbench-plus-workloads --repo-type dataset
```

Upload the contents of `data/external_hdf5/` only:

```powershell
$hfRepo = 'YOUR_HF_USER/bigvectorbench-plus-workloads'
$hfData = '<LOCAL_STAGING>\data\external_hdf5'
hf upload $hfRepo $hfData . --repo-type dataset --commit-message 'Add five BigVectorBench+ workload datasets'
```

Before uploading, verify that `$hfData` contains exactly five `.hdf5` files. Do not upload `data/manifests/` to this repository; those manifests belong in GitHub.

## GitHub

Clone the GitHub repository into a separate directory and create a release branch:

```powershell
$githubUrl = 'https://github.com/OWNER/REPOSITORY.git'
$githubRoot = '<GITHUB_CLONE_PATH>'
git clone $githubUrl $githubRoot
git -C $githubRoot switch -c selective-release-20260921
```

Copy only the selected release paths:

```powershell
$stage = '<LOCAL_STAGING>'
$paths = @(
  'README.md', 'configs', 'docs', 'figures', 'manifests',
  'data\manifests', 'results\build_time',
  'results\diagnostics\ACORN_single_thread_20260920',
  'results\diagnostics\U1',
  'results\raw\correlation_campaign_20260813'
)

foreach ($rel in $paths) {
  $src = Join-Path $stage $rel
  $dst = Join-Path $githubRoot $rel
  if (Test-Path -LiteralPath $src -PathType Container) {
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Copy-Item -Path (Join-Path $src '*') -Destination $dst -Recurse -Force
  } elseif (Test-Path -LiteralPath $src) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Copy-Item -LiteralPath $src -Destination $dst -Force
  }
}

Copy-Item -LiteralPath (Join-Path $stage '.gitignore') -Destination (Join-Path $githubRoot '.gitignore') -Force
```

Check that no HDF5 or internal audit file is staged:

```powershell
git -C $githubRoot status --short
git -C $githubRoot add README.md .gitignore configs docs figures manifests data\manifests results\build_time results\diagnostics\ACORN_single_thread_20260920 results\diagnostics\U1 results\raw\correlation_campaign_20260813
git -C $githubRoot diff --cached --stat
git -C $githubRoot diff --cached --name-only | Select-String '\.hdf5|_remote_stage'
```

The last command must return no output. Then commit and push the branch:

```powershell
git -C $githubRoot commit -m 'Add selective BigVectorBench+ release artifacts'
git -C $githubRoot push -u origin selective-release-20260921
```

Open a pull request from `selective-release-20260921` to the default branch after reviewing the file list. The current selected GitHub files are below ordinary GitHub's 100 MB single-file limit; the HDF5 files are intentionally excluded from GitHub.
