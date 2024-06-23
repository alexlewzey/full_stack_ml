# hadolint global ignore=DL4001,SC1091,DL3008
FROM python:3.12-bullseye

WORKDIR /workspaces/setup

# Install nvm, node, npm and cdk
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | /bin/bash && \
    export NVM_DIR="$HOME/.nvm" && \
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && \
    [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion" && \
    nvm install 22.3.0 && \
    npm install -g aws-cdk@2.147.0

# Install python dependencies
COPY .pre-commit-config.yaml pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry==1.8.3 && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

# Install pre-commit hooks
RUN git init . && \
    pre-commit install --install-hooks

# Install aws cli (update aarch64 to x86_64 if not using ARM)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install


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
RUN wget -O --progress=dot:giga /tmp/hadolint https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64 && \
    mv /tmp/hadolint /usr/local/bin/hadolint && \
    chmod +x /usr/local/bin/hadolint
