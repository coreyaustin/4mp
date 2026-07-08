# 4mp

Poetry-based Python application scaffold for 4MP simulation code.

## Project Structure

```text
.
|-- pyproject.toml
|-- README.md
|-- scripts/
|   |-- README.md
|   |-- dev/
|   |   `-- smoke_optics.py
|   |-- validation/
|   |   `-- beam_focus_check.py
|   `-- data/
|       `-- README.md
|-- src/
|   `-- fourmp/
|       |-- __init__.py
|       |-- __main__.py
|       `-- cli.py
`-- tests/
		`-- test_cli.py
```

## Prerequisites

- Python 3.12+
- Poetry installed: https://python-poetry.org/docs/#installation

## Getting Started

1. Install dependencies:

	 ```bash
	 poetry install
	 ```

2. Run the app:

	 ```bash
	 poetry run 4mp
	 ```

3. Run tests:

	 ```bash
	 poetry run pytest
	 ```

## Common Commands

- Add a dependency:

	```bash
	poetry add <package>
	```

- Add a dev dependency:

	```bash
	poetry add --group dev <package>
	```

- Run linting:

	```bash
	poetry run ruff check .
	```

## Scripts Workflow

- Put quick experiments in `scripts/dev/`.
- Put repeatable scenario checks in `scripts/validation/`.
- Put small script input files in `scripts/data/`.
- Keep reusable logic in `src/fourmp/` and import it from scripts.

Run script examples:

```bash
poetry run python scripts/dev/smoke_optics.py
poetry run python scripts/validation/beam_focus_check.py
```

## Next Steps

- Add simulation modules under `src/fourmp/`.
- Add unit and integration tests under `tests/`.
- Expand CLI commands as the project grows.
