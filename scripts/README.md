# Scripts

This folder is for runnable scripts while the optical system functionality is being built.

## Layout

- `dev/`: quick experiments and one-off checks during development.
- `validation/`: repeatable validation scripts with stable inputs and outputs.
- `data/`: small local files used by scripts (sample configs, fixtures, cached outputs).

## Guidelines

- Keep core logic in `src/fourmp/` and import it from scripts.
- Treat scripts as thin entrypoints.
- Promote stable script logic into package modules and tests.

## Run Examples

```bash
poetry run python scripts/dev/smoke_optics.py
poetry run python scripts/validation/beam_focus_check.py
```