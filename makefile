install:
	rm -rf .venv/
	poetry install

lint:
	pre-commit run --all-files

test:
	pre-commit run --all-files
	python -m pytest tests/ -v
	docker compose -f tests/api/docker-compose.yaml up --build --abort-on-container-exit

ui:
	uvicorn src.api.ui:app --host 0.0.0.0 --port 8080 --reload

deploy:
	cdk bootstrap
	cdk deploy --require-approval never

test-deployment:
	python -m tests.api.validate_deployment

train:
	PYTHONWARNINGS="ignore" python -m src.train.train --file configs/default_config.json

stage:
	python -m src.train.stage_model

mlflow:
	mlflow ui --host 0.0.0.0 --port 8080

get-url:
	aws cloudformation describe-stacks --stack-name CatVsDogStack --query "Stacks[0].Outputs[?OutputKey=='ApiGatewayUrl'].OutputValue" --output text
