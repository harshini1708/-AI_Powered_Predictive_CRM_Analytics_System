#!/bin/bash

# Docker Initialization Script for I-LEAD AMS
# Usage: ./docker-init.sh [development|production]

set -e

ENVIRONMENT=${1:-development}
ENV_FILE=".env.local"

echo "═══════════════════════════════════════════════════════════════"
echo "I-LEAD AMS Docker Initialization"
echo "Environment: $ENVIRONMENT"
echo "═══════════════════════════════════════════════════════════════"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
  echo "❌ Docker is not installed. Please install Docker first."
  exit 1
fi

if ! command -v docker-compose &> /dev/null; then
  echo "❌ Docker Compose is not installed. Please install Docker Compose first."
  exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Create environment file if it doesn't exist
if [ ! -f "$ENV_FILE" ]; then
  echo ""
  echo "📝 Creating environment file: $ENV_FILE"
  cp .env.docker "$ENV_FILE"
  echo "⚠️  Please edit $ENV_FILE with your configuration"
  echo "   Then run this script again."
  exit 0
fi

# Build images
echo ""
echo "🔨 Building Docker images..."
if [ "$ENVIRONMENT" = "production" ]; then
  docker-compose -f docker-compose.prod.yml build
else
  docker-compose build
fi

# Start services
echo ""
echo "🚀 Starting services..."
if [ "$ENVIRONMENT" = "production" ]; then
  docker-compose -f docker-compose.prod.yml up -d
else
  docker-compose up -d
fi

# Wait for database to be ready
echo ""
echo "⏳ Waiting for database to be ready..."
if [ "$ENVIRONMENT" = "production" ]; then
  docker-compose -f docker-compose.prod.yml exec -T db pg_isready -U postgres > /dev/null
else
  docker-compose exec db pg_isready -U postgres > /dev/null
fi

sleep 3

# Run migrations
echo ""
echo "🔄 Running database migrations..."
if [ "$ENVIRONMENT" = "production" ]; then
  docker-compose -f docker-compose.prod.yml exec api npm run migrate
  echo ""
  echo "✅ Production setup complete!"
  echo ""
  echo "Services:"
  docker-compose -f docker-compose.prod.yml ps
else
  docker-compose exec api npm run migrate

  # Seed test data for development
  echo ""
  echo "🌱 Seeding test data..."
  docker-compose exec api npm run seed

  echo ""
  echo "✅ Development setup complete!"
  echo ""
  echo "Services:"
  docker-compose ps

  echo ""
  echo "📌 Quick Commands:"
  echo "   View logs:     docker-compose logs -f api"
  echo "   Run tests:     docker-compose exec api npm test"
  echo "   Shell access:  docker-compose exec api sh"
  echo "   Stop services: docker-compose down"
  echo ""
  echo "API is available at: http://localhost:3000"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
