FROM public.ecr.aws/lambda/python:3.12

WORKDIR "${LAMBDA_TASK_ROOT}"
COPY pyproject.toml poetry.lock ./
RUN pip install poetry==1.8.3 --no-cache-dir && \
    poetry config virtualenvs.create false && \
    poetry install --only main,lambda --no-interaction --no-ansi --no-root

COPY api "${LAMBDA_TASK_ROOT}"

CMD [ "app.handler" ]
