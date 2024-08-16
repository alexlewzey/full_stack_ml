# Specifies the lambda image used to containerise the api/ui.
FROM public.ecr.aws/lambda/python:3.12-arm64

WORKDIR "${LAMBDA_TASK_ROOT}"
COPY pyproject.toml poetry.lock ./
RUN pip install poetry==1.8.3 --no-cache-dir && \
    poetry config virtualenvs.create false && \
    poetry install --only main,ui --no-interaction --no-ansi --no-root

COPY src "${LAMBDA_TASK_ROOT}/src"
COPY configs "${LAMBDA_TASK_ROOT}/configs"


CMD [ "src.api.ui.handler" ]
