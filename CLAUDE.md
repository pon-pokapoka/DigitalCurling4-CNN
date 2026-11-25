# DigitalCurling4-CNN — notes for Claude Code

A C++/LibTorch client for DigitalCurling4: MCTS search over a CNN policy
and value. Build and run instructions are in README.md.

## Working in this repo

- **Branches.** Branch from `development`, and open PRs against
  `development`. Never push straight to `development` or `main`.
- **Signed commits.** Every commit is signed. In the dev container,
  signing uses the host's SSH agent (`.devcontainer/setup-signing.sh`);
  run `.devcontainer/setup-signing.sh --check` before committing. If
  signing isn't available, stop and say so; never pass `--no-gpg-sign`
  or `-c commit.gpgsign=false`.
- **Public repo.** Commit nothing specific to one deployment: hostnames,
  registry addresses, cluster namespaces or sizes (GPU slot counts),
  home paths. Those go in the uncommitted `.devcontainer/local.env` or
  the environment.
- **English only**, including README sections and comments, even where
  the existing text is Japanese. Leave existing Japanese text alone
  unless asked.
- **No binaries or data.** Never commit build outputs, model weights,
  images or training data (`model/` is gitignored). A release is the
  source plus the model files only, with no binary or image: those would
  redistribute third-party code and need a notices file first.
- **No raw passwords on disk**, not even in a gitignored file such as
  `.devcontainer/local.env`: it holds settings only. Logins live in the
  cluster; pass them in memory for one run.
- **The dev container works for anyone.** By default it needs no SSH
  key, GitHub login or Claude; personal options go through the
  uncommitted `.devcontainer/local.env` (see `local.env.example`).
  Dev-environment helpers go in `.devcontainer/`.
- **Memory before builds.** Builds compete with self-play for the same
  machine's memory. Check `free -h` before `cmake --build`, tests or
  `docker build`; if memory is short, say so rather than start a build
  that thrashes.

## Deliberate choices — don't "fix" them

- `config::nLoop` (1024) is larger than `nBatchSize` (512) on purpose:
  headroom for duplicated tracks.
- The search simulates with `fcv1light` for speed, even when a match
  uses `fcv1`.
- The board image's `dpi` is a tunable hyperparameter, kept coarse for
  speed.
- `win_table.csv` is recomputed by the training loop, not by hand.

Compare search changes on unique tracks, depth and visit concentration,
not raw evaluation counts; a strength claim needs a self-play A/B.

## Licences

The engine code from the client template's example
(`src/engine/cnn_engine.*`) is MIT © UEC Takeshi Ito Laboratory and keeps
that notice. Parts from DigitalCurling3-ClientExamples are Unlicense.
