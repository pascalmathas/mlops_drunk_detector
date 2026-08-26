.PHONY: test
test:
	coverage run -m pytest -v -p no:warnings . && coverage report --rcfile=.coveragerc

.PHONY: format
format:
	ruff format . --line-length 120

.PHONY: build
build:
	docker build . -t drunk-detector:latest -f infra/build/dockerfile

.PHONY: up
up: build
	docker compose --file infra/docker-compose.yaml down
	docker compose --file infra/docker-compose.yaml up app