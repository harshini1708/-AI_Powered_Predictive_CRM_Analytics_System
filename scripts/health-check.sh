#!/bin/bash

# Health Check Script
# Monitors system health and alerts if issues detected

FRONTEND_URL="${FRONTEND_URL:-http://localhost:3000}"
BACKEND_URL="${BACKEND_URL:-http://localhost:3001}"
MONGODB_HOST="${MONGODB_HOST:-localhost}"
MONGODB_PORT="${MONGODB_PORT:-27017}"
ALERT_EMAIL="${ALERT_EMAIL:-}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "═══════════════════════════════════════════════════════════════"
echo "I-LEAD AMS Health Check"
echo "═══════════════════════════════════════════════════════════════"
echo "[$(date)] Starting health check..."
echo ""

# Check frontend
echo "🌐 Frontend: $FRONTEND_URL"
if curl -sf "$FRONTEND_URL" > /dev/null 2>&1; then
  echo -e "${GREEN}✓ Frontend is UP${NC}"
  FRONTEND_STATUS=1
else
  echo -e "${RED}✗ Frontend is DOWN${NC}"
  FRONTEND_STATUS=0
fi
echo ""

# Check backend API
echo "⚙️  Backend API: $BACKEND_URL"
if curl -sf "$BACKEND_URL/api/health" > /dev/null 2>&1; then
  echo -e "${GREEN}✓ Backend is UP${NC}"
  BACKEND_STATUS=1
else
  echo -e "${RED}✗ Backend is DOWN${NC}"
  BACKEND_STATUS=0
fi
echo ""

# Check MongoDB
echo "🗄️  MongoDB: $MONGODB_HOST:$MONGODB_PORT"
if mongosh --host "$MONGODB_HOST:$MONGODB_PORT" --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
  echo -e "${GREEN}✓ MongoDB is UP${NC}"
  DB_STATUS=1
else
  echo -e "${RED}✗ MongoDB is DOWN${NC}"
  DB_STATUS=0
fi
echo ""

# Docker status
echo "🐳 Docker Services"
if command -v docker &> /dev/null; then
  RUNNING=$(docker-compose ps -q | wc -l)
  echo "Services running: $RUNNING"
  docker-compose ps --format "table {{.Service}}\t{{.Status}}"
fi
echo ""

# System resources
echo "📊 System Resources"
if command -v free &> /dev/null; then
  echo "Memory Usage:"
  free -h | grep "Mem"
fi

if command -v df &> /dev/null; then
  echo "Disk Usage:"
  df -h | grep -E "/$|vda1"
fi
echo ""

# Summary
echo "═══════════════════════════════════════════════════════════════"
TOTAL=$((FRONTEND_STATUS + BACKEND_STATUS + DB_STATUS))
echo "Overall Status: $TOTAL/3 services UP"

if [ $TOTAL -eq 3 ]; then
  echo -e "${GREEN}✓ All systems operational${NC}"
  EXIT_CODE=0
elif [ $TOTAL -ge 2 ]; then
  echo -e "${YELLOW}⚠ Some services degraded${NC}"
  EXIT_CODE=1
else
  echo -e "${RED}✗ Critical systems down${NC}"
  EXIT_CODE=2
fi

echo "[$(date)] Health check completed."
echo "═══════════════════════════════════════════════════════════════"

# Send alert if configured and issues detected
if [ $EXIT_CODE -ne 0 ] && [ ! -z "$ALERT_EMAIL" ]; then
  echo "[$(date)] Sending alert email to $ALERT_EMAIL..."
  # Alert logic would be added here
fi

exit $EXIT_CODE
