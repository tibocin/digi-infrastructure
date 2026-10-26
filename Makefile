# -----------------------------------------------------------------------------
# File: Makefile
# Purpose: Common infrastructure operations
# Related: docker-compose.yml
# Tags: make, operations, infrastructure
# -----------------------------------------------------------------------------

# Private stack uses explicit -f so override (host ports) is not auto-merged.
PRIVATE_COMPOSE := -f docker-compose.yml -f docker-compose.prod.yml

.PHONY: help up down restart logs ps health private-up private-down private-health assert-private-ports backup restore clean init status

help: ## Show this help message
	@echo "Digi Infrastructure Management"
	@echo "============================="
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

up: ## Start with local override (publishes ports)
	docker compose up -d

down: ## Stop infrastructure services (default project)
	docker compose down

restart: ## Restart infrastructure services
	docker compose restart

logs: ## Show logs from all services
	docker compose logs -f

ps: ## Show running containers
	docker compose ps

health: ## Health checks (honors COMPOSE_FILE; default private pair)
	./scripts/health-check.sh

private-up: ## Start PCS + stores with NO host port publish
	docker compose $(PRIVATE_COMPOSE) up -d postgres neo4j qdrant redis pcs

private-down: ## Stop private-stack compose project
	docker compose $(PRIVATE_COMPOSE) down

private-health: ## Health via compose exec on private network
	./scripts/health-check.sh

assert-private-ports: ## Fail if PCS/data-store ports are published
	./scripts/assert-private-ports.sh

backup: ## Create backup
	docker compose exec backup-sidecar restic backup /backup

restore: ## Restore from backup (latest)
	docker compose exec backup-sidecar restic restore latest --target /restore

clean: ## Remove all containers and volumes
	docker compose $(PRIVATE_COMPOSE) down -v || true
	docker compose down -v
	docker system prune -f

init: ## Initialize databases for all apps
	docker compose exec postgres psql -U digi -d digi -c "SELECT 1;"

status: ## Show detailed status
	@echo "=== Container Status ==="
	@docker compose ps
	@echo ""
	@echo "=== Database Status ==="
	@docker compose exec -T postgres psql -U digi -d digi -c "\l" || echo "PostgreSQL not accessible"
	@echo ""
	@echo "=== Network Status ==="
	@docker network ls | grep digi || echo "No digi networks found"

setup: ## Initial setup (local digi with published ports)
	@echo "Setting up Digi Infrastructure..."
	@if [ ! -f .env ]; then \
		echo "Creating .env from template..."; \
		cp config/env.example .env; \
		echo "Please edit .env with your configuration values"; \
	else \
		echo ".env already exists"; \
	fi
	@echo "Starting infrastructure (local override)..."
	@make up
	@echo "Running health checks..."
	@COMPOSE_FILE=docker-compose.yml:docker-compose.override.yml make health
	@echo "Setup complete. For unpublished ports on a public host, use: make private-up"

dev: ## Start with development overrides (published ports)
	docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

prod: ## Alias for private-up (production / public-host)
	@$(MAKE) private-up

test: ## Run compose contract tests + optional live health
	@uv run --with pytest --with pyyaml pytest tests/test_private_compose.py -q
	@echo "Compose contract tests passed."

update: ## Update all images to latest versions
	docker compose $(PRIVATE_COMPOSE) pull
	docker compose $(PRIVATE_COMPOSE) up -d

monitor: ## Open monitoring interfaces (local digi only)
	@echo "Monitoring UIs require make up / make digi (published ports)."
	@echo "  Grafana / Prometheus / Neo4j Browser — localhost only when override is loaded."


