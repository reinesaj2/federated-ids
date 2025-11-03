# Contributing Guide

Thanks for your interest in improving the Federated IDS project! Follow the checklist below to ensure smooth reviews.

## Workflow

1. **Issue First** – align with an open issue or create one describing the change.
2. **Branch Naming** – use `<type>/<issue>-<slug>` (e.g., `fix/issue-57-aggregation-comparison`).
3. **Worktrees** – create a dedicated worktree per issue to keep contexts isolated.
4. **TDD** – add or update tests before implementing production code.
5. **Commits** – follow Conventional Commits (e.g., `feat(server): add secure aggregation toggle`).

## Development Environment

- Install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  pip install -r requirements-dev.txt
  ```
- Install pre-commit hooks:
  ```bash
  pre-commit install
  ```

## Checks Before Opening a PR

Run the following locally (or use the `make` shortcuts):

```bash
pre-commit run --all-files
pytest --cov=. --cov-report=term-missing
```

Ensure coverage remains above the required threshold and that new functionality is tested.

## Code Style & Typing

- Black enforces formatting (line length 140, string normalization disabled).
- Ruff enforces linting (E, F, I, UP, B, W rule sets).
- Mypy runs with strict settings for aggregation and privacy modules. Add types rather than suppress warnings.

## Documentation & Artifacts

- Update docs under `docs/` when behavior or experiments change.
- Attach relevant plots or metrics when touching experiment pipelines.

## Pull Requests

- Fill out the PR template.
- Link the related issue (`Fixes #<number>`).
- Summarize test coverage and CI status.

We appreciate your contributions!
