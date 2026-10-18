# Free-only GitHub Actions setup for the football ATS model

## What it does

- On a manual run, installs the **free** `nflreadpy` library and downloads actual nflverse 2026 play-by-play, schedule, and weekly roster datasets.
- Rejects empty/malformed inputs, saves compressed CSV data, records a manifest with SHA-256 hashes, collection timestamps, row counts, and retrieval methods.
- Runs the existing unit tests and verifies the downloaded files.
- Optionally replays **2026-10-18 Texans at Jaguars**, excluding any plays from the target game or other games that could not have completed by the 2026-10-18 12:30 UTC cutoff.
- Marks a replay as incomplete when historical pregame publication, player status, pressure, odds prices, or additional inputs cannot be verified.
- Uploads raw source snapshots and model audits as GitHub Actions artifacts that can be downloaded from the run's **Artifacts** section, with 30-day retention.
- Automatically collects one weekly Sunday snapshot, at 11:15 UTC. Scheduled execution can be delayed; actual collection timestamps, not the schedule time, determine pregame eligibility.

No billing information, paid API keys, subscriptions, or commerce tools are required for this design. Use a **public repository** for GitHub-hosted Actions without consuming private-repository Actions quotas (subject to GitHub's then-current free usage conditions).

## One-time action required from the repository owner

1. Open https://github.com/new and create a **public**, empty repository, suggested name `football-ats-model` (do not initialize with a README, so it is easier to upload the project).
2. Grant your connected GitHub app access to this repository if necessary (GitHub account Settings > Applications > Installed GitHub Apps > Configure).
3. Return to ChatGPT with the new `owner/repository` name. The assistant can then publish the Python model and workflows using the connected GitHub integration.
4. In GitHub open **Actions** > **Free nflverse collection and replay** > **Run workflow**. Choose `season` 2026 and leave `replay_texans_jaguars` on to create a retrospective data audit.
5. Open the completed run. Read the run summary, then download the resulting **nflverse-[run number]** artifact. An incomplete historical replay is **not** a validated prediction.

## Source and license caution

nflverse source feeds can have different licenses. Confirm the current license before republishing the datasets. The workflow stores collection artifacts under your own Actions run; it does not automatically commit large data files into Git history or publish model inputs as a commercial product.

## Limitations

- This repository has not yet been executed on GitHub-hosted infrastructure until the first workflow run succeeds.
- `nflreadpy` and nflverse CSV schemas may change. The schema checks will then fail visibly instead of silently making assumptions.
- An October 18, 2026 **postgame** download is a retrospective dataset, never proof of an archived pre-kickoff snapshot. Opponent games are still strictly filtered before the forecast cutoff.
- The free-only pipeline will not guess unverified pressure/coverage, betting-price history, medical availability, or personnel effects. If they remain missing, the strict model blocks a full game simulation and the audit explains why.
- GitHub Actions artifacts here expire after **30 days**. Download runs you need for long-term history. Permanent archival can be added separately using only free, appropriate storage options.
- The standalone model's coefficients are provisional and betting win rates have not been validated on historical holdouts.
