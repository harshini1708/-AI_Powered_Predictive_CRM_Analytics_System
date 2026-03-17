# I-LEAD AMS Deployment Guide - Phase 7D Edition

**Version**: 4.0+
**Last Updated**: March 17, 2026
**Status**: Production Ready

---

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone and setup
git clone <repository-url>
cd ilead-ams

# Copy environment templates
cp ilead-ams/backend/.env.example ilead-ams/backend/.env
cp ilead-ams/frontend/.env.example ilead-ams/frontend/.env.local

# Edit with your settings
nano ilead-ams/backend/.env
nano ilead-ams/frontend/.env.local

# Deploy
./scripts/deploy.sh up

# Verify
./scripts/deploy.sh status
```

**Access:**
- Frontend: http://localhost:3000
- Backend: http://localhost:3001
- MongoDB: localhost:27017

---

## 📋 Prerequisites

### System Requirements
- OS: Linux (Ubuntu 20.04+), macOS, Windows WSL2
- CPU: 2+ cores (4+ for production)
- RAM: 4GB minimum (8GB+ recommended)
- Disk: 50GB free space
- Docker 20.10+
- Docker Compose 2.0+
- Git 2.30+

### Install Docker (Ubuntu/Debian)

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

---

## 🏠 Local Development

### 1. Clone & Install

```bash
git clone <repository-url>
cd ilead-ams

# Backend
cd ilead-ams/backend && npm install

# Frontend
cd ../frontend && npm install
cd ../..
```

### 2. Configure Environment

```bash
cp ilead-ams/backend/.env.example ilead-ams/backend/.env
cp ilead-ams/frontend/.env.example ilead-ams/frontend/.env.local

# Edit .env files
nano ilead-ams/backend/.env
nano ilead-ams/frontend/.env.local
```

### 3. Start Services

**Option A - Docker Compose:**
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

**Option B - Manual:**
```bash
# Terminal 1: Backend
cd ilead-ams/backend
npm run dev

# Terminal 2: Frontend
cd ilead-ams/frontend
npm start

# Terminal 3: MongoDB
mongod
```

### 4. Verify

```bash
curl http://localhost:3000           # Frontend
curl http://localhost:3001/api/health # Backend
mongosh localhost:27017              # MongoDB
```

---

## 🌐 Production Deployment

### 1. Server Setup

**Recommended:** Ubuntu 20.04 LTS on:
- AWS EC2 (t3.medium or larger)
- DigitalOcean (4GB RAM, 80GB SSD)
- Google Cloud Platform
- Azure VMs

### 2. Initial Configuration

```bash
ssh -i your-key.pem ubuntu@your-server-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
sudo apt install -y git

# Clone repository
git clone <repository-url>
cd ilead-ams
```

### 3. Environment Setup

```bash
cp ilead-ams/backend/.env.example ilead-ams/backend/.env
cp ilead-ams/frontend/.env.example ilead-ams/frontend/.env.local

nano ilead-ams/backend/.env
nano ilead-ams/frontend/.env.local
```

**Critical Production Values:**

```env
# Backend .env
NODE_ENV=production
JWT_SECRET=<run: openssl rand -hex 32>
MONGODB_URI=mongodb://ilead_admin:strong_password@mongodb:27017/ilead_ams?authSource=admin
CORS_ORIGIN=https://your-domain.com
FRONTEND_URL=https://your-domain.com

# Email Setup
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=<app-specific-password>
SMTP_FROM=noreply@your-domain.com

# Frontend .env.local
REACT_APP_API_URL=https://api.your-domain.com
REACT_APP_ENV=production
```

### 4. SSL Certificates

```bash
# Install Certbot
sudo apt install -y certbot

# Generate certificate
sudo certbot certonly --standalone -d your-domain.com -d api.your-domain.com

# Copy to project
sudo mkdir -p /etc/ilead-ams/ssl
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem /etc/ilead-ams/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem /etc/ilead-ams/ssl/key.pem
sudo chown -R ubuntu:ubuntu /etc/ilead-ams
```

### 5. Deploy

```bash
./scripts/deploy.sh up
./scripts/deploy.sh status
./scripts/health-check.sh
```

### 6. DNS Configuration

Update DNS records:
```
A record:  your-domain.com    → your-server-ip
A record:  api.your-domain.com → your-server-ip (optional, routed via nginx)
CNAME:     www.your-domain.com → your-domain.com
```

---

## 🐳 Docker Management

### Core Services

```
Services Started:
├── mongodb:6.0-alpine      (Port 27017)
├── backend (Node.js 18)    (Port 3001)
├── frontend (React+Nginx)  (Port 3000)
├── redis:7-alpine          (Port 6379)
└── nginx:alpine            (Ports 80, 443)
```

### Common Commands

```bash
# Start all services
docker-compose up -d

# View status
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f mongodb

# Stop services
docker-compose down

# Restart service
docker-compose restart backend

# Execute command
docker-compose exec backend npm run migrate

# Remove all data (destructive)
docker-compose down -v
```

---

## 💾 Database Management

### Automated Backups

```bash
# Create backup
./scripts/backup-database.sh

# View backups
ls -lh ./backups/

# Schedule daily backups at 2 AM
echo "0 2 * * * /home/ubuntu/ilead-ams/scripts/backup-database.sh" | crontab -
```

### Restore from Backup

```bash
# List backups
ls ./backups/backup_*.archive

# Restore
./scripts/restore-database.sh ./backups/backup_20260317_120000.archive

# Verify
docker-compose exec mongodb mongosh -u ilead_admin -p
> use ilead_ams
> db.hourLogs.countDocuments()
```

### Database Maintenance

```bash
# Connect to MongoDB
docker-compose exec mongodb mongosh -u ilead_admin -p

# Check size
> db.stats()

# Reindex
> db.users.reIndex()
> db.hourLogs.reIndex()

# Clean old notifications (30+ days)
> db.notifications.deleteMany({
    createdAt: {$lt: new Date(Date.now() - 30*24*60*60*1000)}
  })
```

---

## 📊 Monitoring & Health Checks

### Health Status

```bash
# Run health check
./scripts/health-check.sh

# Automated checks every 5 minutes
echo "*/5 * * * * /home/ubuntu/ilead-ams/scripts/health-check.sh" | crontab -
```

### View Logs

```bash
# Backend logs
docker-compose logs -f --tail=100 backend

# Frontend logs
docker-compose logs -f --tail=100 frontend

# Database logs
docker-compose logs -f --tail=100 mongodb

# All logs
docker-compose logs -f
```

### Performance Monitoring

```bash
# Real-time resource usage
docker stats

# Container details
docker inspect ilead-backend

# Network stats
docker network inspect ilead-network
```

---

## 🔧 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs <service>

# Rebuild
docker-compose build --no-cache <service>

# Restart
docker-compose restart <service>
```

### Database Connection Issues

```bash
# Verify MongoDB is running
docker-compose ps mongodb

# Test connection
docker-compose exec mongodb mongosh -u ilead_admin -p

# Check MONGODB_URI in .env
cat ilead-ams/backend/.env | grep MONGODB_URI
```

### Frontend Shows Blank Page

```bash
# Clear browser cache
# Check frontend logs
docker-compose logs frontend

# Test API connection
curl http://localhost:3001/api/health
```

### Email Not Sending

```bash
# Check logs
docker-compose logs backend | grep -i email

# Test SMTP
docker-compose exec backend node -e "
const nodemailer = require('nodemailer');
const transporter = nodemailer.createTransport({
  host: process.env.SMTP_HOST,
  port: process.env.SMTP_PORT,
  auth: {user: process.env.SMTP_USER, pass: process.env.SMTP_PASS}
});
transporter.verify((err, ok) => console.log(err || 'SMTP OK'));
"
```

---

## 🔐 Security Checklist

### Before Deployment
- [ ] Generate new JWT_SECRET: `openssl rand -hex 32`
- [ ] Change all default passwords
- [ ] Set NODE_ENV=production
- [ ] Remove debug logging
- [ ] Enable HTTPS
- [ ] Configure CORS to your domain
- [ ] Run `npm audit` on both projects
- [ ] All secrets in .env (not committed)

### During Deployment
- [ ] Strong database password (20+ characters)
- [ ] MongoDB authentication enabled
- [ ] Firewall configured (only ports 22, 80, 443 open)
- [ ] HTTPS enforced via nginx
- [ ] Non-root user running containers
- [ ] Read-only file systems where possible

### After Deployment
- [ ] Test HTTPS only
- [ ] Verify rate limiting works
- [ ] Check logs for errors
- [ ] Test email notifications
- [ ] Verify backups work
- [ ] Monitor for suspicious activity

### Firewall (UFW)

```bash
sudo ufw enable
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw status
```

---

## 📅 Maintenance Schedule

**Daily:** Health checks, Monitor error logs, Verify backups

**Weekly:** Disk space, Security updates, Database integrity

**Monthly:** Update images, Review logs, Performance analysis

**Quarterly:** Security audit, Load testing, Disaster recovery

---

## 🚀 Deployment Scripts

```bash
# Deploy/Start services
./scripts/deploy.sh up

# Stop services
./scripts/deploy.sh down

# View logs
./scripts/deploy.sh logs [service]

# Health check
./scripts/deploy.sh status
./scripts/health-check.sh

# Database backup
./scripts/backup-database.sh

# Database restore
./scripts/restore-database.sh ./backups/backup_xxx.archive

# Update deployment
./scripts/deploy.sh update

# Rollback
./scripts/deploy.sh rollback
```

---

## 📞 Support

**Documentation:**
- Deployment: This file
- Architecture: README.md
- API: API Documentation
- Database: DATABASE_SCHEMA.md

**Monitoring Tools:**
- DataDog, New Relic, ELK Stack, Prometheus

**Cloud Platforms:**
- AWS, Google Cloud, Azure, DigitalOcean, Heroku

---

## 📝 Version History

| Version | Date | Notes |
|---------|------|-------|
| 4.0 | 2026-03-17 | Phase 7D - Docker & Production Ready |
| 3.5 | 2026-03-10 | Monitoring & Health Checks |
| 3.0 | 2026-02-15 | Initial Docker Setup |

**Status**: ✅ Production Ready
**Last Updated**: March 17, 2026

