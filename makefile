install:
	rm -rf .venv/
	poetry install

lint:
	pre-commit run --all-files

test:
	poetry lock
	pre-commit run --all-files
	python -m pytest tests/ -v
	docker compose -f tests/api/docker-compose.yaml up --build --abort-on-container-exit

ui:
	uvicorn src.api.ui:app --host 0.0.0.0 --port 8080 --reload

deploy:
	cdk bootstrap
	cdk deploy --require-approval never

train:
	python -m src.train.train

stage:
	python -m src.train.stage_model

mlflow:
	mlflow ui --host 0.0.0.0 --port 8080
