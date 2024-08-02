#! /bin/bash
poetry install
pre-commit install --install-hooks
git config --global --add safe.directory $(pwd)
