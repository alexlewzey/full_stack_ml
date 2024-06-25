# hadolint global ignore=DL4001,SC1091,DL3008,DL3016,DL3047
FROM python:3.12-bullseye

WORKDIR /workspaces/setup

# Install node, npm, cdk and taplo (toml formatter)
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs awscli && \
    npm install -g aws-cdk@2.147.0 @taplo/cli

# Install python dependencies
COPY .pre-commit-config.yaml pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry==1.8.3 && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

# Install pre-commit hooks
RUN git init . && \
    pre-commit install --install-hooks

# Update and install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common \
    gnupg2 && \
    rm -rf /var/lib/apt/lists/*

# Install Docker
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    echo "deb [arch=arm64] https://download.docker.com/linux/debian $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install --no-install-recommends -y docker-ce docker-ce-cli containerd.io && \
    rm -rf /var/lib/apt/lists/*

# Install hadolint (docker linter)
RUN wget -O /tmp/hadolint https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64 && \
    mv /tmp/hadolint /usr/local/bin/hadolint && \
    chmod +x /usr/local/bin/hadolint
