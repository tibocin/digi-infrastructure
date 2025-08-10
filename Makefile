# -----------------------------------------------------------------------------
# File: Makefile
# Purpose: Common infrastructure operations
# Related: docker-compose.yml
# Tags: make, operations, infrastructure
# -----------------------------------------------------------------------------

.PHONY: help up down restart logs ps health backup restore clean init status

help: ## Show this help message
	@echo "Digi Infrastructure Management"
	@echo "============================="
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

up: ## Start infrastructure services
	docker-compose up -d

down: ## Stop infrastructure services
	docker-compose down

restart: ## Restart infrastructure services
	docker-compose restart

logs: ## Show logs from all services
	docker-compose logs -f

ps: ## Show running containers
	docker-compose ps

health: ## Run health checks
	./scripts/health-check.sh

backup: ## Create backup
	docker-compose exec backup-sidecar restic backup /backup

restore: ## Restore from backup (latest)
	docker-compose exec backup-sidecar restic restore latest --target /restore

clean: ## Remove all containers and volumes
	docker-compose down -v
	docker system prune -f

init: ## Initialize databases for all apps
	docker-compose exec postgres psql -U digi -d digi -c "SELECT 1;"

status: ## Show detailed status
	@echo "=== Container Status ==="
	@docker-compose ps
	@echo ""
	@echo "=== Database Status ==="
	@docker-compose exec -T postgres psql -U digi -d digi -c "\l" || echo "PostgreSQL not accessible"
	@echo ""
	@echo "=== Network Status ==="
	@docker network ls | grep digi || echo "No digi networks found"

setup: ## Initial setup and configuration
	@echo "Setting up Digi Infrastructure..."
	@if [ ! -f .env ]; then \
		echo "Creating .env from template..."; \
		cp config/env.example .env; \
		echo "Please edit .env with your configuration values"; \
	else \
		echo ".env already exists"; \
	fi
	@echo "Starting infrastructure..."
	@make up
	@echo "Running health checks..."
	@make health
	@echo "Setup complete! Access services at:"
	@echo "  - Grafana: http://localhost:3000"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Neo4j Browser: http://localhost:7474"
	@echo "  - ChromaDB: http://localhost:8001"

dev: ## Start with development overrides
	docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

prod: ## Start with production configuration
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

test: ## Run infrastructure tests
	@echo "Testing infrastructure components..."
	@make health
	@echo "All tests passed!"

update: ## Update all images to latest versions
	docker-compose pull
	docker-compose up -d

monitor: ## Open monitoring interfaces
	@echo "Opening monitoring interfaces..."
	@if command -v open >/dev/null 2>&1; then \
		open http://localhost:3000; \
		open http://localhost:9090; \
		open http://localhost:7474; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open http://localhost:3000; \
		xdg-open http://localhost:9090; \
		xdg-open http://localhost:7474; \
	else \
		echo "Please open manually:"; \
		echo "  Grafana: http://localhost:3000"; \
		echo "  Prometheus: http://localhost:9090"; \
		echo "  Neo4j Browser: http://localhost:7474"; \
	fi


