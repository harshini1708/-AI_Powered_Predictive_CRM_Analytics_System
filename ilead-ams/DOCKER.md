# Docker Setup for I-LEAD AMS

This guide provides instructions for building and running the I-LEAD AMS system using Docker.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+

## Project Structure

```
ilead-ams/
├── docker-compose.yml          # Development configuration
├── docker-compose.prod.yml     # Production configuration
├── .env.docker                 # Docker environment template
├── backend/
│   ├── Dockerfile              # Multi-stage build for Node.js app
│   └── .dockerignore           # Files to exclude from build
└── ... (other directories)
```

## Development Setup

### 1. Prepare Environment

Copy the Docker environment template:

```bash
cp .env.docker .env.local
```

Edit `.env.local` with your configuration:

```env
# Database
DB_USER=postgres
DB_PASSWORD=yourpassword
DB_NAME=ilead_ams

# JWT
JWT_SECRET=your-secure-random-secret-32-chars-minimum

# Email (optional for development)
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
```

### 2. Build and Start Services

```bash
# Build images and start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Wait for database to be ready (~10-15 seconds)
docker-compose ps
```

The API will be available at `http://localhost:3000`

### 3. Run Database Migrations

```bash
# Run migrations
docker-compose exec api npm run migrate

# Seed test data
docker-compose exec api npm run seed
```

### 4. Run Tests

```bash
# Run all tests
docker-compose exec api npm test

# Watch mode
docker-compose exec api npm run test:watch

# Coverage report
docker-compose exec api npm test -- --coverage
```

### 5. Development Workflow

The backend service runs with `npm run dev` (nodemon) for auto-reload:

```bash
# Make code changes in your editor
# Changes will automatically reload in the container

# View live logs
docker-compose logs -f api

# Access container shell if needed
docker-compose exec api sh
```

### 6. Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Production Setup

### 1. Prepare Environment

Create `.env.prod` with production values:

```bash
cp .env.docker .env.prod
```

**Critical production settings:**

```env
NODE_ENV=production
DB_PASSWORD=generate-strong-random-password
JWT_SECRET=generate-strong-random-secret
SUPER_ADMIN_PIN_HASH=bcrypt-hash-of-admin-pin
STORAGE_TYPE=s3  # Use AWS S3 for file storage
AWS_S3_BUCKET=your-production-bucket
```

### 2. Build Production Image

```bash
# Build without running
docker-compose -f docker-compose.prod.yml build

# Or rebuild if you modify code
docker-compose -f docker-compose.prod.yml build --no-cache
```

### 3. Start Production Services

```bash
# Start in production mode
docker-compose -f docker-compose.prod.yml up -d

# View logs (last 100 lines)
docker-compose -f docker-compose.prod.yml logs -f --tail=100 api
```

### 4. Database Management

```bash
# Create backup
docker-compose -f docker-compose.prod.yml exec db \
  pg_dump -U postgres ilead_ams > backup.sql

# Restore backup
docker-compose -f docker-compose.prod.yml exec -T db \
  psql -U postgres ilead_ams < backup.sql

# Run migrations (if needed after update)
docker-compose -f docker-compose.prod.yml exec api npm run migrate
```

### 5. Monitor Production

```bash
# Health check status
docker-compose -f docker-compose.prod.yml ps

# View resource usage
docker stats

# Check application health
curl http://localhost:3000/health
```

### 6. Update Production

```bash
# Pull latest code
git pull

# Rebuild image
docker-compose -f docker-compose.prod.yml build

# Restart services (zero-downtime if using container orchestration)
docker-compose -f docker-compose.prod.yml up -d
```

## Image Details

### Backend Image (Dockerfile)

- **Base**: `node:18-alpine` (lightweight, ~160MB)
- **Multi-stage build**: Separates dependencies from runtime
- **User**: Runs as non-root `nodejs` user (security)
- **Health check**: Validates service availability
- **Exposed**: Port 3000

Image size: ~300MB (production)

### Database Image

- **Base**: `postgres:15-alpine` (lightweight)
- **Persistence**: Volume-mounted data directory
- **Network**: Private Docker network
- **Port**: 5432 (localhost only in production)

## Networking

Services communicate via Docker network bridge:

- **Development**: `ilead-network`
- **Production**: `ilead-network-prod`

Backend can reach database at `db:5432`

## Volume Management

### Development Volumes

- `postgres_data`: Database files (persistent)
- `api_uploads`: Uploaded documents (persistent, synced with local)
- Node modules: Anonymous volume for performance

### Production Volumes

- `postgres_data_prod`: Database backup point
- `api_uploads_prod`: Production file storage (persist between deployments)

## Troubleshooting

### Database Connection Issues

```bash
# Check database is running
docker-compose ps db

# View database logs
docker-compose logs db

# Test connection
docker-compose exec api npm run migrate:latest
```

### Port Already in Use

If port 3000 is in use:

```bash
# Change in docker-compose.yml
API_PORT=3001 docker-compose up

# Or modify .env.local
echo "API_PORT=3001" >> .env.local
```

### Out of Memory

```bash
# Increase Docker resources in preferences
# Then restart:
docker-compose down && docker-compose up -d
```

### Permission Issues

```bash
# Fix upload directory permissions
docker-compose exec api chmod 755 uploads

# Or rebuild without volume caching
docker-compose down -v && docker-compose build --no-cache
```

## Environment Variables Reference

See `.env.docker` for complete list with descriptions.

### Key Variables

| Variable | Development | Production | Required |
|----------|-------------|-----------|----------|
| `NODE_ENV` | development | production | Yes |
| `DB_HOST` | db | db | Yes |
| `DB_USER` | postgres | postgres | Yes |
| `DB_PASSWORD` | postgres | *** | Yes |
| `JWT_SECRET` | dev-key | *** | Yes |
| `STORAGE_TYPE` | local | s3 | Yes |
| `SMTP_USER` | (optional) | *** | No |
| `CLEARSTREAM_API_KEY` | (optional) | *** | No |

## Security Considerations

### Development

- ⚠️ Uses default credentials (postgres/postgres)
- ⚠️ Debug logging enabled
- ✅ Isolated Docker network

### Production

- ✅ Strong credentials required
- ✅ Non-root user execution
- ✅ Health checks enabled
- ✅ Logging to files (not stdout)
- ✅ Database on private network only
- ✅ S3 for file storage (not local filesystem)
- ✅ TLS/SSL for database connections

## Performance Tuning

### Development

Run fewer concurrent services if memory-constrained:

```bash
# Start only database and API (skip other services)
docker-compose up -d db api
```

### Production

Database optimizations in `docker-compose.prod.yml`:

```env
-c shared_buffers=256MB -c max_connections=200
```

Adjust based on:
- Number of concurrent users
- Available server memory
- Expected query patterns

## Deployment Examples

### AWS ECS

Push image to ECR:

```bash
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag ilead-ams-api <account>.dkr.ecr.<region>.amazonaws.com/ilead-ams:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/ilead-ams:latest
```

### Kubernetes

See `k8s/` directory for Kubernetes manifests.

### Docker Swarm

```bash
docker stack deploy -c docker-compose.prod.yml ilead-ams
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Node.js in Docker Best Practices](https://nodejs.org/en/docs/guides/nodejs-docker-webapp/)
- [PostgreSQL Docker Hub](https://hub.docker.com/_/postgres)

## Support

For issues:

1. Check logs: `docker-compose logs api`
2. Verify configuration: `docker-compose config`
3. Review Docker documentation
4. Submit issue with logs to repository
