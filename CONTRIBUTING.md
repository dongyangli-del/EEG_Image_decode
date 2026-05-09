# Contributing to EEG Image Decode

Thank you for your interest in contributing!  This project is a research codebase accompanying the NeurIPS 2024 paper *"Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion"*.

## Ways to Contribute

- **Bug reports** — File an issue using the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).
- **Feature requests** — File an issue using the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).
- **Pull requests** — See the guidelines below.
- **Questions** — Open a [Discussion](../../discussions) rather than an issue.

## Development Setup

1. Fork and clone the repository.
2. Create a conda environment:
   ```bash
   . setup.sh
   conda activate BCI
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```

## Code Style

- **Python** — Follow [PEP 8](https://peps.python.org/pep-0008/).  Line length ≤ 100 characters.
- **Docstrings** — Use Google-style docstrings for public functions.
- **Comments** — English only.
- **Type hints** — Preferred for new public APIs.

## Pull Request Checklist

Before submitting a PR please ensure:

- [ ] Code runs without errors (`python3 -m py_compile <file.py>`).
- [ ] All existing benchmark scripts still run correctly.
- [ ] New features are documented in the PR description.
- [ ] If you add a dependency, update `requirements.txt` *and* `setup.sh`.
- [ ] Commit messages are descriptive (e.g. `fix: correct subject-token slicing in iTransformer`).

## Reporting Issues

When filing a bug report, please include:

1. Python and PyTorch versions.
2. Full traceback / error message.
3. Minimal command to reproduce the issue.
4. Dataset split and subject ID if relevant.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
