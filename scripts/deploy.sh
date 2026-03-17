#!/bin/bash

# Deployment Script
# Manages Docker deployment, updates, and rollback

set -e

ACTION="${1:-help}"
COMPOSE_FILE="docker-compose.yml"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "I-LEAD AMS Deployment Manager"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

deploy_up() {
  echo "[$(date)] Starting services..."
  docker-compose -f "$COMPOSE_FILE" up -d
  echo "[$(date)] Services started. Waiting for health checks..."
  sleep 10
  docker-compose -f "$COMPOSE_FILE" ps
}

deploy_down() {
  echo "[$(date)] Stopping services..."
  docker-compose -f "$COMPOSE_FILE" down
  echo "[$(date)] Services stopped."
}

deploy_logs() {
  SERVICE="${2:-}"
  if [ -z "$SERVICE" ]; then
    docker-compose -f "$COMPOSE_FILE" logs -f --tail=100
  else
    docker-compose -f "$COMPOSE_FILE" logs -f --tail=100 "$SERVICE"
  fi
}

deploy_status() {
  echo "[$(date)] Checking service status..."
  docker-compose -f "$COMPOSE_FILE" ps
  echo ""
  echo "Health Checks:"
  curl -s http://localhost/health || echo "Frontend: ❌"
  curl -s http://localhost/api/health || echo "Backend: ❌"
}

deploy_backup() {
  echo "[$(date)] Creating database backup before deployment..."
  ./scripts/backup-database.sh
}

deploy_update() {
  echo "[$(date)] Pulling latest changes..."
  git pull origin main
  
  echo "[$(date)] Building images..."
  docker-compose -f "$COMPOSE_FILE" build
  
  echo "[$(date)] Backing up database..."
  ./scripts/backup-database.sh
  
  echo "[$(date)] Updating containers..."
  docker-compose -f "$COMPOSE_FILE" up -d
  
  echo "[$(date)] Update completed. Checking status..."
  sleep 5
  deploy_status
}

deploy_restart() {
  echo "[$(date)] Restarting services..."
  docker-compose -f "$COMPOSE_FILE" restart
  echo "[$(date)] Services restarted."
}

deploy_clean() {
  echo "[$(date)] WARNING: This will remove all containers and volumes!"
  read -p "Continue? (yes/no): " -n 3 -r
  echo
  if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    docker-compose -f "$COMPOSE_FILE" down -v
    echo "[$(date)] Cleanup completed."
  else
    echo "Cleanup cancelled."
  fi
}

deploy_rollback() {
  echo "[$(date)] Rolling back to previous version..."
  git revert --no-edit HEAD
  echo "[$(date)] Restoring database from latest backup..."
  LATEST_BACKUP=$(ls -t ./backups/backup_*.archive 2>/dev/null | head -n 1)
  if [ -n "$LATEST_BACKUP" ]; then
    ./scripts/restore-database.sh "$LATEST_BACKUP"
  fi
  echo "[$(date)] Restarting services..."
  docker-compose -f "$COMPOSE_FILE" restart
}

case "$ACTION" in
  up)
    deploy_up
    ;;
  down)
    deploy_down
    ;;
  logs)
    deploy_logs "$@"
    ;;
  status)
    deploy_status
    ;;
  restart)
    deploy_restart
    ;;
  backup)
    deploy_backup
    ;;
  update)
    deploy_update
    ;;
  clean)
    deploy_clean
    ;;
  rollback)
    deploy_rollback
    ;;
  *)
    echo "Usage: $0 {up|down|logs|status|restart|backup|update|clean|rollback|help}"
    echo ""
    echo "Commands:"
    echo "  up           - Start all services"
    echo "  down         - Stop all services"
    echo "  logs [svc]   - View logs (optional service filter)"
    echo "  status       - Check service status"
    echo "  restart      - Restart all services"
    echo "  backup       - Backup database"
    echo "  update       - Update and deploy latest version"
    echo "  clean        - Remove all containers and volumes"
    echo "  rollback     - Rollback to previous version"
    echo "  help         - Show this help message"
    echo ""
    exit 1
    ;;
esac
