# Production Deployment Guide

This guide covers deploying and maintaining the I-LEAD AMS in production environments.

## Table of Contents

- [Pre-Deployment](#pre-deployment)
- [Environment Setup](#environment-setup)
- [Deployment Methods](#deployment-methods)
- [Database Management](#database-management)
- [Monitoring](#monitoring)
- [Scaling](#scaling)
- [Disaster Recovery](#disaster-recovery)
- [Security Hardening](#security-hardening)
- [Troubleshooting](#troubleshooting)

## Pre-Deployment

### Pre-Deployment Checklist

- [ ] Code reviewed and merged to main
- [ ] All tests passing (80%+ coverage)
- [ ] Security scan passed (no high/critical issues)
- [ ] Docker image built and tested
- [ ] Release notes prepared
- [ ] Database migration tested on staging
- [ ] Backup plan documented
- [ ] Rollback plan prepared
- [ ] Team notified of deployment window
- [ ] Documentation updated

### Environment Requirements

**Minimum Requirements**:
- CPU: 2 cores
- RAM: 4GB
- Storage: 50GB
- Uptime SLA: 99.5%

**Recommended Setup**:
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 100GB+
- Uptime SLA: 99.9%

**Staging Environment**:
- Match production as closely as possible
- Test all deployments here first
- Use production-like data volumes
- Monitor performance metrics

### Required Services

- Docker Engine 20.10+
- Docker Compose 2.0+
- PostgreSQL 15+
- Redis (optional, for caching)
- Nginx or similar reverse proxy

### Domain and SSL

```bash
# Generate SSL certificate (Let's Encrypt)
sudo certbot certonly --standalone -d api.example.com

# Store in secure location
/etc/letsencrypt/live/api.example.com/
  ├── cert.pem
  ├── chain.pem
  ├── fullchain.pem
  └── privkey.pem
```

## Environment Setup

### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

### 2. Application Directory

```bash
# Create application directory
sudo mkdir -p /app/ilead-ams
cd /app/ilead-ams

# Set permissions
sudo chown $USER:$USER .
chmod 755 .

# Create required subdirectories
mkdir -p uploads backups logs
chmod 755 uploads logs
```

### 3. Environment Configuration

```bash
# Copy production environment template
cp .env.docker .env.prod

# Edit configuration
nano .env.prod
```

**Critical Production Settings**:

```env
# Core
NODE_ENV=production
PORT=3000
LOG_LEVEL=info

# Database (strong credentials!)
DB_HOST=db
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=generate-strong-password-here-32-chars-min
DB_NAME=ilead_ams
DB_SSL=true

# JWT Secrets (generate with: node -e "console.log(require('crypto').randomBytes(32).toString('hex'))")
JWT_SECRET=your-random-32-char-secret-key-here-change-me
SUPER_ADMIN_PIN_HASH=bcrypt-hash-of-your-admin-pin

# Email Configuration (use SendGrid for reliability)
SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USER=apikey
SMTP_PASS=your-sendgrid-api-key-here
SMTP_FROM=I-LEAD AMS <noreply@i-leadusa.org>

# Storage (use AWS S3 for production)
STORAGE_TYPE=s3
AWS_REGION=us-east-1
AWS_S3_BUCKET=ilead-ams-prod
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key

# SMS (Clearstream)
CLEARSTREAM_API_KEY=your-clearstream-api-key

# Security
CORS_ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com
FRONTEND_URL=https://app.example.com
```

**Generate Strong Secrets**:

```bash
# JWT Secret
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"

# Database Password
openssl rand -base64 32

# Admin PIN Hash (assume PIN is 071676)
node -e "const bcrypt = require('bcryptjs'); console.log(bcrypt.hashSync('071676', 10))"
```

### 4. SSL/TLS Setup

```bash
# Create SSL directory
mkdir -p ssl
cp /etc/letsencrypt/live/api.example.com/fullchain.pem ssl/
cp /etc/letsencrypt/live/api.example.com/privkey.pem ssl/

# Set permissions
chmod 644 ssl/fullchain.pem
chmod 600 ssl/privkey.pem

# Auto-renewal setup
sudo systemctl enable certbot-renew
sudo systemctl start certbot-renew.timer
```

## Deployment Methods

### Method 1: GitHub Actions (Automated - Recommended)

**Setup**:
1. Configure deployment secrets in GitHub
2. Create GitHub Release with version tag
3. Workflow runs automatically

**Triggers**:
- Release published
- Manual workflow dispatch

**Process**:
```bash
# Release automatically triggers deploy workflow
# - Pulls latest code
# - Runs migrations
# - Deploys containers
# - Health checks
```

### Method 2: Manual Deployment

**Prerequisites**:
```bash
# Login to server
ssh deploy@api.example.com

# Navigate to app directory
cd /app/ilead-ams
```

**Deployment Steps**:

```bash
# 1. Pull latest code
git fetch origin
git checkout v1.2.3  # Use version tag

# 2. Update environment if needed
nano .env.prod

# 3. Build images
docker-compose -f docker-compose.prod.yml build

# 4. Backup database (before migrations!)
docker-compose -f docker-compose.prod.yml exec db \
  pg_dump -U postgres ilead_ams > backups/ilead_ams_$(date +%Y%m%d_%H%M%S).sql

# 5. Start services
docker-compose -f docker-compose.prod.yml up -d

# 6. Run migrations
docker-compose -f docker-compose.prod.yml exec api npm run migrate

# 7. Verify health
curl https://api.example.com/health

# 8. View logs
docker-compose -f docker-compose.prod.yml logs -f api
```

### Method 3: Blue-Green Deployment

For zero-downtime deployments:

```bash
# 1. Start new version (blue)
docker-compose -f docker-compose.blue.yml up -d

# 2. Run migrations
docker-compose -f docker-compose.blue.yml exec api npm run migrate

# 3. Health check
curl https://api-blue.example.com/health

# 4. Switch traffic (green -> blue)
# Update reverse proxy to point to blue

# 5. Keep green running for rollback
# If issues, switch back to green

# 6. After stable period, tear down green
docker-compose -f docker-compose.green.yml down
```

## Database Management

### Migrations

```bash
# Check migration status
docker-compose -f docker-compose.prod.yml exec api \
  npx knex migrate:status

# Run migrations
docker-compose -f docker-compose.prod.yml exec api \
  npm run migrate

# Rollback last migration
docker-compose -f docker-compose.prod.yml exec api \
  npm run migrate:rollback

# Rollback all
docker-compose -f docker-compose.prod.yml exec api \
  npx knex migrate:rollback --all
```

### Backup Strategy

**Daily Automated Backups**:

```bash
#!/bin/bash
# /usr/local/bin/backup-ilead-ams.sh

BACKUP_DIR="/app/ilead-ams/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/ilead_ams_$TIMESTAMP.sql"

# Backup
docker-compose -f docker-compose.prod.yml exec -T db \
  pg_dump -U postgres ilead_ams > "$BACKUP_FILE"

# Compress
gzip "$BACKUP_FILE"

# Keep last 30 days
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +30 -delete

# Upload to S3
aws s3 cp "$BACKUP_FILE.gz" s3://ilead-ams-backups/$TIMESTAMP/
```

**Cron Job**:

```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * /usr/local/bin/backup-ilead-ams.sh
```

### Restore from Backup

```bash
# Restore from backup file
docker-compose -f docker-compose.prod.yml exec -T db \
  psql -U postgres ilead_ams < backups/ilead_ams_20260316_020000.sql

# Or from S3
aws s3 cp s3://ilead-ams-backups/20260316_020000/ilead_ams_20260316_020000.sql.gz - | \
  gunzip | \
  docker-compose -f docker-compose.prod.yml exec -T db \
  psql -U postgres ilead_ams
```

## Monitoring

### Health Checks

```bash
# API health
curl -s https://api.example.com/health | jq .

# Database connectivity
docker-compose -f docker-compose.prod.yml exec db \
  pg_isready -U postgres
```

### Logging

**View Logs**:

```bash
# Last 100 lines
docker-compose -f docker-compose.prod.yml logs --tail=100 api

# Follow live logs
docker-compose -f docker-compose.prod.yml logs -f api

# Since specific time
docker-compose -f docker-compose.prod.yml logs --since 2h api
```

**Log Rotation**:

```json
// docker-compose.prod.yml
"logging": {
  "driver": "json-file",
  "options": {
    "max-size": "10m",
    "max-file": "5",
    "labels": "service=ilead-ams"
  }
}
```

### Metrics Monitoring

**CPU & Memory**:

```bash
# Real-time stats
docker stats

# Per-container
docker stats --no-stream
```

**Application Metrics**:
- Response time (in logs)
- Error rate (grep for errors)
- Database queries (PostgreSQL logs)

### External Monitoring

**Recommended Services**:
- **APM**: New Relic, DataDog, Elastic APM
- **Logging**: ELK Stack, Splunk, CloudWatch
- **Alerting**: PagerDuty, Opsgenie
- **Status Page**: Statuspage.io

**Setup Example (New Relic)**:

```javascript
// Add to src/server.js
const newrelic = require('newrelic');

// Configure with API key
// NEW_RELIC_LICENSE_KEY=your-license-key
// NEW_RELIC_APP_NAME=ilead-ams
```

## Scaling

### Horizontal Scaling

**Database Replication**:

```bash
# Setup read replica
# Configure in postgresql.conf
wal_level = replica
max_wal_senders = 3

# Promote replica if primary fails
docker-compose -f docker-compose.prod.yml exec replica \
  pg_ctl promote -D /var/lib/postgresql/data
```

**Multiple API Instances**:

```yaml
# docker-compose.prod.yml
services:
  api-1:
    container_name: ilead-ams-api-1
    # ...
  api-2:
    container_name: ilead-ams-api-2
    # ...
  api-3:
    container_name: ilead-ams-api-3
    # ...
```

**Load Balancing**:

```nginx
# /etc/nginx/sites-available/ilead-ams
upstream ilead_ams {
  server api-1:3000;
  server api-2:3000;
  server api-3:3000;
}

server {
  listen 443 ssl http2;
  server_name api.example.com;

  ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

  location / {
    proxy_pass http://ilead_ams;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Connection pooling
    proxy_http_version 1.1;
    proxy_set_header Connection "";
  }
}
```

### Vertical Scaling

Increase server resources:
- Add more CPU cores
- Increase RAM
- Use faster storage (SSD)
- Optimize database indexes

## Disaster Recovery

### Runbook

**API Crash**:
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs api

# Restart
docker-compose -f docker-compose.prod.yml restart api

# If persistent, check database
docker-compose -f docker-compose.prod.yml exec db psql -U postgres -c "SELECT 1"
```

**Database Crash**:
```bash
# Restore from backup
docker-compose -f docker-compose.prod.yml down
# Restore backup
docker-compose -f docker-compose.prod.yml up -d
docker-compose -f docker-compose.prod.yml exec api npm run migrate
```

**Disk Full**:
```bash
# Check usage
df -h

# Clean Docker artifacts
docker system prune -a

# Archive old logs
tar czf logs_archive_$(date +%Y%m%d).tar.gz logs/
rm -rf logs/*
```

### RTO/RPO Targets

- **RTO** (Recovery Time Objective): 1 hour
- **RPO** (Recovery Point Objective): 15 minutes

Achieved through:
- Automated hourly backups
- Replication to different zone
- Docker image in registry
- Infrastructure as Code

## Security Hardening

### Network Security

```bash
# Firewall rules
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw enable

# Limit SSH
sudo nano /etc/ssh/sshd_config
# - PermitRootLogin no
# - PasswordAuthentication no
# - PubkeyAuthentication yes
```

### Access Control

```bash
# Create deploy user
sudo useradd -m -s /bin/bash deploy

# Add SSH key
sudo mkdir -p ~deploy/.ssh
sudo cp your_key.pub ~deploy/.ssh/authorized_keys
sudo chown -R deploy:deploy ~deploy/.ssh
sudo chmod 700 ~deploy/.ssh

# Sudo privileges
echo "deploy ALL=(ALL) NOPASSWD: /usr/bin/docker, /usr/local/bin/docker-compose" | \
  sudo tee /etc/sudoers.d/deploy
```

### Secrets Management

**DO NOT**:
- Commit secrets to git
- Share in chat/email
- Use default passwords
- Log sensitive data

**DO**:
- Use environment variables
- Rotate secrets regularly
- Use secrets manager (HashiCorp Vault, AWS Secrets Manager)
- Audit access logs

### Database Security

```sql
-- Create limited user
CREATE USER api_user WITH PASSWORD 'strong-password';

-- Grant minimal permissions
GRANT USAGE ON SCHEMA public TO api_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO api_user;

-- No superuser
```

## Troubleshooting

### Common Issues

**High Memory Usage**:
```bash
# Check top processes
docker stats

# Restart container
docker-compose -f docker-compose.prod.yml restart api

# Check for memory leaks
docker logs api | grep -i error
```

**Slow Database Queries**:
```bash
# Enable query logging
docker-compose -f docker-compose.prod.yml exec db \
  psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"

# Check slow queries
docker-compose -f docker-compose.prod.yml exec db \
  psql -U postgres -c "SELECT * FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

**Connection Pool Issues**:
```bash
# Check connections
docker-compose -f docker-compose.prod.yml exec db \
  psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# Increase pool size
# In docker-compose.prod.yml increase max_connections
```

**Disk Space**:
```bash
# Check usage
docker system df

# Clean up
docker system prune --volumes
docker image prune --all --force
```

## Support & Documentation

- [DOCKER.md](./ilead-ams/DOCKER.md) - Docker setup guide
- [CONTRIBUTING.md](./CONTRIBUTING.md) - Development guide
- [API.md](./API.md) - API documentation
- GitHub Issues - Report problems
- Pull Requests - Contribute improvements

## Checklist: Post-Deployment

- [ ] All health checks passing
- [ ] Logs monitored for errors
- [ ] Backups completed successfully
- [ ] Performance metrics within acceptable range
- [ ] Team notified of successful deployment
- [ ] Release notes published
- [ ] Documentation updated
- [ ] Monitoring dashboards configured
- [ ] Incident response plan reviewed
- [ ] Schedule post-deployment review meeting

## Rollback Procedure

If issues occur:

```bash
# 1. Identify issue
docker-compose -f docker-compose.prod.yml logs api | head -50

# 2. Stop current deployment
docker-compose -f docker-compose.prod.yml down

# 3. Revert to previous version
git checkout v1.2.2

# 4. Restore previous database backup
docker-compose -f docker-compose.prod.yml up -d db
# Restore backup

# 5. Start services with previous version
docker-compose -f docker-compose.prod.yml up -d

# 6. Verify health
curl https://api.example.com/health

# 7. Document incident and analyze root cause
```

---

**Last Updated**: March 2026
**Maintained By**: I-LEAD Development Team
**Version**: 1.0.0
