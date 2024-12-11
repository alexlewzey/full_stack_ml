# hadolint global ignore=DL3013
# Specifies the lambda image used to containerise the api/ui.
FROM public.ecr.aws/lambda/python:3.12-arm64

ARG DAGSHUB_USER_TOKEN
ENV DAGSHUB_USER_TOKEN=$DAGSHUB_USER_TOKEN

WORKDIR "${LAMBDA_TASK_ROOT}"
COPY pyproject.toml uv.lock ./
RUN pip install uv --no-cache-dir && \
    uv venv && \
    uv sync --group ui


COPY src "${LAMBDA_TASK_ROOT}/src"
COPY configs "${LAMBDA_TASK_ROOT}/configs"
RUN uv run python -m src.train.download_champion

CMD [ "src.api.ui.handler" ]
