install:
	rm -rf .venv/
	poetry install

test:
	@pre-commit run --all-files

e2e:
	@docker compose -f tests/api/docker-compose.yaml up --build --abort-on-container-exit

ui-run:
	@uvicorn src.api.ui:app --host 0.0.0.0 --port 8080 --reload

app-deploy:
	cdk bootstrap
	cdk deploy --require-approval never

train:
	@python -m src.train.train

train-monitor:
	@mlflow ui --host 0.0.0.0 --port 8080
