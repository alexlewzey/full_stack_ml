FROM python:3.12-bullseye

WORKDIR /workspaces/setup

# Install python dependencies
COPY .pre-commit-config.yaml pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root && \
    git init . && \
    pre-commit install --install-hooks
