install:
	uv sync --all-extras
	uv run pre-commit install

install-no-pre-commit:
	uv pip install ".[dev]"

fix:
	uv run pre-commit run --all-files

test:
	uv run pytest --cov=PACKAGE --cov-report=term-missing
