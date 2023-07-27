POETRY=poetry run
PROJECT=qod-ai

## help: print this help message
.PHONY: help
help:
	@echo 'Usage:'
	@sed -n 's/^##//p' ${MAKEFILE_LIST} | column -t -s ':' |  sed -e 's/^/ /'

setup:
	asdf plugin-add python || true
	asdf plugin-add poetry https://github.com/asdf-community/asdf-poetry.git || true
	asdf install
	poetry install
	$(POETRY) pre-commit install

## ruff: run the ruff linter
.PHONY: ruff
ruff:
	$(POETRY) ruff qod bin tests

## mypy: run the mypy linter
.PHONY: mypy
mypy:
	$(POETRY) mypy --ignore-missing-imports --install-types --non-interactive .

## lint: run all linters
.PHONY: lint
lint: ruff mypy

## test: run all linters and then test project
.PHONY: test
test: lint
	$(POETRY) python -m pytest tests

## server: run the app API in server.py using uvicorn
.PHONY: server
server:
	$(POETRY) uvicorn qod.server:app --reload

.PHONY: cli
cli:
	$(POETRY) python -m bin.cli

.PHONY: cli_summary
cli_summary:
	$(POETRY) python -m bin.cli_summary
