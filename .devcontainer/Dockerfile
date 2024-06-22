FROM python:3.12-bullseye

WORKDIR /workspaces/setup

# Install python dependencies
COPY .pre-commit-config.yaml pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root && \
    git init . && \
    pre-commit install --install-hooks

# Install nvm, node, npm and cdk
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && \
    source ~/.bashrc && \
    nvm install 22.3.0 && \
    npm install -g aws-cdk