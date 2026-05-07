COMPOSE_FILE := infra/docker-compose.yml

.PHONY: up down logs status build restart test pull ps up-future

up:
	docker compose -f $(COMPOSE_FILE) up -d

up-future:
	docker compose -f $(COMPOSE_FILE) --profile future up -d

down:
	docker compose -f $(COMPOSE_FILE) down

logs:
	docker compose -f $(COMPOSE_FILE) logs -f $(service)

status:
	docker compose -f $(COMPOSE_FILE) ps

ps: status

build:
	docker compose -f $(COMPOSE_FILE) build

pull:
	docker compose -f $(COMPOSE_FILE) pull

restart:
	docker compose -f $(COMPOSE_FILE) restart $(service)

test:
	python -m pytest
