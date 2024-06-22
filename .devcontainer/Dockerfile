FROM python:3.12-bullseye

WORKDIR /workspaces/setup

# Install nvm, node, npm and cdk
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | /bin/bash && \
    export NVM_DIR="$HOME/.nvm" && \
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && \
    [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion" && \
    nvm install 22.3.0 && \
    npm install -g aws-cdk

# Install python dependencies
COPY .pre-commit-config.yaml pyproject.toml poetry.lock ./
RUN pip install poetry && \
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
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common \
    gnupg2

# Install Docker
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    echo "deb [arch=arm64] https://download.docker.com/linux/debian $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y docker-ce docker-ce-cli containerd.io

