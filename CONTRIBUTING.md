# Contributing

## Development setup

```bash
git clone https://github.com/d7rectCE/data-drift-fdr
cd data-drift-fdr
pip install -e ".[dev]"
pytest
```

## Conventions

- Public functions and classes need docstrings. `docs/api.md` is generated from them:
  run `python docs/gen_api.py` after changing the public API (`tests/test_docs.py` fails otherwise).
- A change in the method's behaviour needs evidence: an experiment in `experiments/` (the setting
  goes in the module docstring, tables go to `results/`) and a section in `docs/experiments.md`
  and `docs/experiments.ru.md`.
- Documentation is kept in English and Russian (`*.ru.md`); the API reference is English only.
- Integrations import their third-party package only when used, so the core keeps its small set of
  dependencies.

## Releasing

1. Update `version` in `pyproject.toml` and add a section to `CHANGELOG.md`.
2. Commit, then create a GitHub release with a tag `vX.Y.Z`.
3. The `publish` workflow builds the package and uploads it to PyPI through trusted publishing.

One-time setup of trusted publishing, by the owner of the PyPI project: on pypi.org, under
"Publishing", add a trusted publisher for GitHub with owner `d7rectCE`, repository
`data-drift-fdr`, workflow `publish.yml` and environment `pypi`; in the GitHub repository create
an environment named `pypi` (Settings → Environments). No token is stored in the repository.
