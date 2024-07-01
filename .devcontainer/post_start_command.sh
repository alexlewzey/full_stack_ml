poetry install
poetry run pre-commit install --install-hooks
git config --global --add safe.directory $(pwd)
