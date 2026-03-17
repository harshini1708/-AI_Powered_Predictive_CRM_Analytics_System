# Phase 2: Production Deployment Guide

**Status**: Ready for Execution 🚀
**Estimated Duration**: 1-2 hours
**Prerequisites**: Domain name, server access, SMTP credentials

---

## 📋 Quick Start (TL;DR)

If you already have a server with Docker installed:

```bash
# 1. Clone and checkout branch
git clone https://github.com/your-org/-AI_Powered_Predictive_CRM_Analytics_System.git
cd -AI_Powered_Predictive_CRM_Analytics_System
git checkout claude/migrate-nodejs-backend-ISFDE

# 2. Configure environment
cp ilead-ams/backend/.env.example ilead-ams/backend/.env
cp ilead-ams/frontend/.env.example ilead-ams/frontend/.env.local
# ✏️ Edit both .env files with your values

# 3. Build and deploy
docker-compose build
./scripts/deploy.sh up

# 4. Verify
./scripts/health-check.sh
```

---

## 🎯 Step-by-Step Deployment

### STEP 1: Server Provisioning

**Option A: AWS EC2**
```bash
# Launch Ubuntu 20.04 LTS instance
# - Instance type: t3.medium or larger (4GB RAM)
# - Storage: 80GB SSD minimum
# - Security group: Allow 22 (SSH), 80 (HTTP), 443 (HTTPS)

# Connect to server
ssh -i your-key.pem ubuntu@your-server-ip
```

**Option B: Other Provider (DigitalOcean, Linode, etc.)**
- Choose Ubuntu 20.04 LTS
- 4GB RAM, 80GB SSD
- Configure firewall to allow ports 22, 80, 443

**On Server: System Setup**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install git and tools
sudo apt install -y git curl wget unzip

# Configure firewall
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
```

### STEP 2: Domain & SSL Setup

**Configure DNS**
1. Purchase domain (or use existing)
2. Point A records:
   - `your-domain.com` → your-server-ip
   - `api.your-domain.com` → your-server-ip
3. Wait for DNS propagation (5-30 minutes)

**Generate SSL Certificates**
```bash
# On server
sudo apt install -y certbot

# Generate certificate
sudo certbot certonly --standalone \
  -d your-domain.com \
  -d api.your-domain.com \
  -d www.your-domain.com \
  --non-interactive \
  --agree-tos \
  -m admin@your-domain.com
```

**Make certificates accessible**
```bash
sudo mkdir -p /opt/ilead-ams/ssl
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem /opt/ilead-ams/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem /opt/ilead-ams/ssl/key.pem
sudo chown -R $USER:$USER /opt/ilead-ams
```

### STEP 3: Clone Repository

```bash
# On server
cd /opt
git clone https://github.com/your-org/-AI_Powered_Predictive_CRM_Analytics_System.git
cd -AI_Powered_Predictive_CRM_Analytics_System
git checkout claude/migrate-nodejs-backend-ISFDE
```

### STEP 4: Environment Configuration

**Backend Configuration**
```bash
# Create .env file
cp ilead-ams/backend/.env.example ilead-ams/backend/.env

# Edit with production values
nano ilead-ams/backend/.env
```

**Required Environment Variables:**
```env
# Core
NODE_ENV=production
PORT=3001
MONGODB_URI=mongodb://admin:YOUR_SECURE_PASSWORD@mongodb:27017/ilead_ams?authSource=admin

# Security
JWT_SECRET=$(openssl rand -hex 32)
JWT_EXPIRE=7d

# URLs
FRONTEND_URL=https://your-domain.com
CORS_ORIGIN=https://your-domain.com
API_BASE_URL=https://api.your-domain.com

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
SMTP_FROM=noreply@your-domain.com
SMTP_SECURE=false

# Optional: S3 for file uploads
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key-id
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_S3_BUCKET=ilead-ams-prod

# Logging
LOG_LEVEL=info
LOG_DIR=./logs
```

**Frontend Configuration**
```bash
# Create .env.local
cp ilead-ams/frontend/.env.example ilead-ams/frontend/.env.local

# Edit values
nano ilead-ams/frontend/.env.local
```

**Frontend Environment Variables:**
```env
REACT_APP_API_URL=https://api.your-domain.com
REACT_APP_ENV=production
REACT_APP_ENABLE_ANALYTICS=true
REACT_APP_ENABLE_ERROR_TRACKING=true
```

**Docker Compose Secrets**
```bash
# Edit docker-compose.yml to set MongoDB credentials
nano docker-compose.yml
```

Update the MongoDB service section:
```yaml
services:
  mongodb:
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: your-secure-password
      MONGO_INITDB_DATABASE: ilead_ams
```

### STEP 5: Update Nginx Configuration

```bash
# Edit Nginx config with your domain
nano docker-compose.yml
```

Find the nginx service and update the environment:
```yaml
nginx:
  environment:
    - DOMAIN_NAME=your-domain.com
    - API_DOMAIN=api.your-domain.com
    - SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
    - SSL_KEY_PATH=/etc/nginx/ssl/key.pem
```

### STEP 6: Build & Deploy

```bash
# Validate build structure
./validate-build.sh
# Should show: ✅ ALL VALIDATIONS PASSED (46/46)

# Build Docker images
docker-compose build

# Start all services
./scripts/deploy.sh up
# OR manually:
# docker-compose up -d

# Wait for services to start (2-3 minutes)
docker-compose ps
```

### STEP 7: Database Initialization

```bash
# Run migrations
docker-compose exec backend npm run migrate

# Seed initial data (optional, for testing)
docker-compose exec backend npm run seed

# Verify database
docker-compose exec mongodb mongosh -u admin -p your-password
> use ilead_ams
> db.users.countDocuments()
> exit
```

### STEP 8: Verification

```bash
# Check system health
./scripts/health-check.sh

# Check all services running
docker-compose ps

# Monitor service logs
docker-compose logs -f backend
docker-compose logs -f nginx

# Test connectivity
curl -I https://your-domain.com
curl -I https://your-domain.com/api/health
```

---

## ✅ POST-DEPLOYMENT VALIDATION

### Functionality Tests
```bash
# 1. Test frontend
# Open https://your-domain.com in browser
# - Page should load
# - Dark/light mode toggle works
# - Responsive on mobile

# 2. Test API health
curl https://your-domain.com/api/health

# 3. Test login (use test credentials)
curl -X POST https://your-domain.com/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@test.com","password":"password"}'

# 4. Test email (trigger notification action)
# - Perform action that sends email
# - Check email inbox within 5 minutes
```

### Performance Checks
```bash
# Monitor resource usage
docker stats

# Expected:
# - CPU: < 20% per service
# - Memory: < 500MB backend, < 200MB frontend
# - Disk: Monitor with: df -h
```

### Security Checks
```bash
# Verify HTTPS redirect
curl -i http://your-domain.com
# Should return 301 with Location: https://

# Check security headers
curl -i https://your-domain.com | grep -i "Strict-Transport\|Content-Security\|X-Frame"

# Should show HSTS, CSP, and other security headers

# Test rate limiting
for i in {1..100}; do curl https://your-domain.com/api/health; done
# After limit: should return 429 Too Many Requests
```

---

## 🔐 Backup Configuration

```bash
# Set up automated backups
crontab -e

# Add daily backup (3 AM UTC)
0 3 * * * /opt/-AI_Powered_Predictive_CRM_Analytics_System/scripts/backup-database.sh

# Or manual backup:
./scripts/backup-database.sh

# Restore from backup:
./scripts/restore-database.sh ./backups/ilead_ams_backup_TIMESTAMP.tar.gz
```

---

## 🆘 Troubleshooting

### Services won't start
```bash
# Check logs
docker-compose logs backend
docker-compose logs mongodb
docker-compose logs nginx

# Common issues:
# - Port already in use: change ports in docker-compose.yml
# - MongoDB connection: check credentials in .env
# - Nginx: check domain/SSL path configuration
```

### Database connection error
```bash
# Verify MongoDB is running
docker-compose ps mongodb

# Check connection string in backend .env
# Format: mongodb://user:password@mongodb:27017/database?authSource=admin

# Test connection from backend
docker-compose exec backend npm run test:db-connection
```

### SSL certificate issues
```bash
# Verify certificate exists
sudo ls -la /opt/ilead-ams/ssl/

# Renew certificate (do 30 days before expiry)
sudo certbot renew --force-renewal

# Copy new certificate
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem /opt/ilead-ams/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem /opt/ilead-ams/ssl/key.pem

# Restart nginx
docker-compose restart nginx
```

### Email not sending
```bash
# Check email service status
curl http://localhost:3001/api/admin/email/status

# Verify SMTP credentials in .env:
# - SMTP_HOST, SMTP_PORT correct
# - SMTP_USER is Gmail email
# - SMTP_PASS is app-specific password (not regular Gmail password)

# Test email manually:
docker-compose exec backend node -e "
const emailService = require('./src/services/emailService.js');
emailService.sendEmail('test@example.com', 'Test', '<h1>Test</h1>')
  .then(r => console.log('Email sent:', r))
  .catch(e => console.error('Error:', e));
"
```

---

## 📊 Monitoring

### Daily Operations
```bash
# Check health every day
./scripts/health-check.sh

# View logs for errors
docker-compose logs --since 24h | grep -i error

# Monitor disk space
df -h /opt

# Monitor backups
ls -lh ./backups/ | tail -5
```

### Weekly Tasks
```bash
# Check SSL certificate expiry
sudo certbot certificates

# Review security logs
docker-compose logs nginx | grep "403\|401" | wc -l

# Backup verification
# Verify latest backup can be restored
```

### Monthly Tasks
```bash
# Performance analysis
docker stats --no-stream

# Database optimization
docker-compose exec mongodb mongosh
> db.users.getIndexes()
> db.hourLogs.getIndexes()

# Update Docker images
docker-compose pull
docker-compose build
```

---

## 🚀 Deployment Complete!

Once all steps are complete and verified:

1. ✅ Domain accessible at https://your-domain.com
2. ✅ API responding at https://your-domain.com/api/health
3. ✅ SSL certificate valid and auto-renewing
4. ✅ Email notifications working
5. ✅ Database backing up daily
6. ✅ Health checks passing

**Next Steps:**
- Monitor for 24 hours for any issues
- Proceed to **Phase 3: Complete Phase 7D Week 3** (RBAC & Performance)
- Begin **Phase 4: Start Phase 7E Mobile App** development

---

**Document Version**: 1.0
**Created**: March 17, 2026
**Status**: Ready for Production Deployment
