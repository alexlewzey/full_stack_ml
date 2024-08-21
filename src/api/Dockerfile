# Specifies the lambda image used to containerise the api/ui.
FROM public.ecr.aws/lambda/python:3.12-arm64

ARG DAGSHUB_USER_TOKEN
ENV DAGSHUB_USER_TOKEN=$DAGSHUB_USER_TOKEN

WORKDIR "${LAMBDA_TASK_ROOT}"
COPY pyproject.toml poetry.lock ./
RUN pip install poetry==1.8.3 --no-cache-dir && \
    poetry config virtualenvs.create false && \
    poetry install --only main,ui --no-interaction --no-ansi --no-root

COPY src "${LAMBDA_TASK_ROOT}/src"
COPY configs "${LAMBDA_TASK_ROOT}/configs"
RUN python -m src.train.download_champion

CMD [ "src.api.ui.handler" ]
