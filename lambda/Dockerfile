FROM public.ecr.aws/lambda/python:3.12

# COPY pyproject.toml poetry.lock ./

# RUN pip install poetry && \
#     poetry config virtualenvs.create false && \
#     poetry install --only main,lambda --no-interaction --no-ansi --no-root


COPY lambda "${LAMBDA_TASK_ROOT}/lambda"

RUN chmod +x "${LAMBDA_TASK_ROOT}/lambda/hello.py"

CMD ["lambda.hello.handler"]
