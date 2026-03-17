.PHONY: help dev prod init up down logs shell test migrate seed build rebuild

# Color output
BLUE=\033[0;34m
GREEN=\033[0;32m
YELLOW=\033[0;33m
NC=\033[0m # No Color

help:
	@echo "$(BLUE)I-LEAD AMS Docker Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@echo "  make init          Initialize development environment"
	@echo "  make dev           Start development services"
	@echo "  make up            Start all services"
	@echo "  make down          Stop all services"
	@echo "  make logs          View live logs"
	@echo "  make shell         Access backend shell"
	@echo "  make test          Run tests"
	@echo "  make migrate       Run database migrations"
	@echo "  make seed          Seed test data"
	@echo ""
	@echo "$(GREEN)Build Commands:$(NC)"
	@echo "  make build         Build Docker images"
	@echo "  make rebuild       Rebuild images without cache"
	@echo ""
	@echo "$(GREEN)Production Commands:$(NC)"
	@echo "  make prod          Start production services"
	@echo "  make prod-down     Stop production services"
	@echo "  make prod-logs     View production logs"
	@echo "  make prod-backup   Backup production database"
	@echo ""

# Development targets
init:
	@echo "$(BLUE)Initializing development environment...$(NC)"
	./docker-init.sh development

dev:
	@echo "$(BLUE)Starting development services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "API: http://localhost:3000"

up: dev

down:
	@echo "$(BLUE)Stopping all services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

logs:
	@echo "$(BLUE)Following logs (Ctrl+C to stop)...$(NC)"
	docker-compose logs -f api

shell:
	@echo "$(BLUE)Accessing backend shell...$(NC)"
	docker-compose exec api sh

test:
	@echo "$(BLUE)Running tests...$(NC)"
	docker-compose exec api npm test

test-watch:
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	docker-compose exec api npm run test:watch

test-coverage:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	docker-compose exec api npm test -- --coverage

migrate:
	@echo "$(BLUE)Running database migrations...$(NC)"
	docker-compose exec api npm run migrate

migrate-rollback:
	@echo "$(BLUE)Rolling back migrations...$(NC)"
	docker-compose exec api npm run migrate:rollback

seed:
	@echo "$(BLUE)Seeding test data...$(NC)"
	docker-compose exec api npm run seed

lint:
	@echo "$(BLUE)Running ESLint...$(NC)"
	docker-compose exec api npm run lint

lint-fix:
	@echo "$(BLUE)Fixing ESLint issues...$(NC)"
	docker-compose exec api npm run lint:fix

# Build targets
build:
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker-compose build
	@echo "$(GREEN)✓ Build complete$(NC)"

rebuild:
	@echo "$(BLUE)Rebuilding Docker images (no cache)...$(NC)"
	docker-compose build --no-cache
	@echo "$(GREEN)✓ Build complete$(NC)"

# Production targets
prod:
	@echo "$(BLUE)Starting production services...$(NC)"
	docker-compose -f docker-compose.prod.yml up -d
	@echo "$(GREEN)✓ Production services started$(NC)"

prod-down:
	@echo "$(BLUE)Stopping production services...$(NC)"
	docker-compose -f docker-compose.prod.yml down
	@echo "$(GREEN)✓ Production services stopped$(NC)"

prod-logs:
	@echo "$(BLUE)Following production logs (Ctrl+C to stop)...$(NC)"
	docker-compose -f docker-compose.prod.yml logs -f api

prod-backup:
	@echo "$(BLUE)Backing up production database...$(NC)"
	@mkdir -p backups
	docker-compose -f docker-compose.prod.yml exec -T db \
		pg_dump -U postgres ilead_ams > backups/ilead_ams_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✓ Backup complete$(NC)"

prod-migrate:
	@echo "$(BLUE)Running migrations in production...$(NC)"
	docker-compose -f docker-compose.prod.yml exec api npm run migrate

# Utility targets
clean:
	@echo "$(YELLOW)Removing stopped containers, networks, and images...$(NC)"
	docker-compose down -v
	docker system prune -f
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-prod:
	@echo "$(YELLOW)Removing production containers and volumes...$(NC)"
	docker-compose -f docker-compose.prod.yml down -v
	@echo "$(GREEN)✓ Production cleanup complete$(NC)"

ps:
	@echo "$(BLUE)Container Status:$(NC)"
	docker-compose ps

ps-prod:
	@echo "$(BLUE)Production Container Status:$(NC)"
	docker-compose -f docker-compose.prod.yml ps

status:
	@echo "$(BLUE)System Status:$(NC)"
	@echo "$(BLUE)Docker:$(NC)"
	docker --version
	@echo "$(BLUE)Docker Compose:$(NC)"
	docker-compose --version
	@echo ""
	@echo "$(BLUE)Services:$(NC)"
	docker-compose ps

health:
	@echo "$(BLUE)Checking service health...$(NC)"
	@curl -s http://localhost:3000/health | jq .
	@echo "$(GREEN)✓ API is healthy$(NC)"
