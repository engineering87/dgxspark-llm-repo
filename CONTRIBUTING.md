# Contributing

Thanks for considering a contribution. This repository is small and focused, so the workflow is light.

## Before you open a PR

1. **Open an issue first** if the change is non-trivial (more than a typo or a small clarification). Discussing the approach saves rework.
2. **Run shellcheck** on any shell script you modify:
   ```bash
   shellcheck scripts/*.sh
   ```
3. **Test on real hardware** when possible. Changes that affect the build flags or the systemd unit should be validated on at least one CUDA-capable Linux box. State explicitly which configuration you tested on in the PR description.
4. **Keep PRs focused.** One concern per PR. Mixing a documentation rewrite with a build flag change makes review difficult.

## Style

- Markdown: one sentence per line in long-form sections is preferred for cleaner diffs.
- Shell: `#!/usr/bin/env bash`, `set -euo pipefail`, lowercase variables for locals, uppercase for environment overrides.
- Commit messages: imperative mood, short subject line, optional body explaining the why.

## Reporting issues

When reporting a build or runtime issue, include:

- Output of `nvidia-smi`
- Output of `nvcc --version`
- Output of `gcc --version`
- The exact command that failed
- Relevant lines from `journalctl -u llama-server` if applicable

## License

By contributing, you agree that your contributions will be licensed under the MIT License of the repository.
