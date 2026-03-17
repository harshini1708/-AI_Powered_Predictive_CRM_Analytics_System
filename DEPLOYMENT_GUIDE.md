# I-LEAD AMS v4 - Deployment Guide
## Apprenticeship Management System with Advanced Analytics

**Version**: V4 (Phase 7C Complete)
**Release Date**: March 2026
**Status**: Production Ready

---

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Deployment Architecture](#deployment-architecture)
3. [Pre-Deployment Checklist](#pre-deployment-checklist)
4. [AWS Lightsail Setup](#aws-lightsail-setup)
5. [Backend Installation](#backend-installation)
6. [Frontend Installation](#frontend-installation)
7. [Database Configuration](#database-configuration)
8. [Environment Variables](#environment-variables)
9. [SSL/TLS Configuration](#ssltls-configuration)
10. [Testing & Verification](#testing--verification)
11. [Troubleshooting](#troubleshooting)
12. [Support & Maintenance](#support--maintenance)

---

## System Overview

### What's Included in V4

**Phase 7C Features** (Latest Release):
- ✅ Advanced Analytics Dashboard with charts
- ✅ Competency heatmap visualization (BACB domains A-F)
- ✅ Domain progress tracking (6 OJT domains)
- ✅ Advanced filtering system for hour logs
- ✅ CSV and PDF export functionality
- ✅ Responsive UI/UX for all screen sizes

**Core Features** (Phases 7A-7B):
- Full user authentication with JWT tokens
- Role-based access control (Apprentice, Supervisor, Administrator, Super Admin)
- Apprentice management and tracking
- Hour log submission and approval workflow
- User administration and permissions
- Mobile-responsive design
- Real-time data validation

### Technology Stack

**Backend**:
- Node.js (>=16.0.0) with Express
- MongoDB for data persistence
- JWT for authentication
- Swagger/OpenAPI for API documentation

**Frontend**:
- React 18.2.0 with hooks
- Vite for fast development and optimized builds
- Recharts for data visualization
- Axios for API calls
- React Router for navigation
- CSS3 with responsive design

**Infrastructure**:
- AWS Lightsail (1GB RAM, 1vCPU minimum recommended)
- Ubuntu 20.04 LTS or newer
- Nginx as reverse proxy
- PM2 for process management

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Client Browser                         │
│                (React SPA - Vite Build)                  │
└────────────────────┬────────────────────────────────────┘
                     │ HTTPS
                     ↓
┌─────────────────────────────────────────────────────────┐
│            Nginx Reverse Proxy (Port 80/443)            │
│                  • SSL/TLS Termination                   │
│                  • Static File Serving                   │
│                  • API Request Routing                   │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP (Internal)
         ┌───────────┴───────────┐
         ↓                       ↓
    ┌────────────┐          ┌──────────┐
    │ Backend    │          │ Backend  │
    │ (PM2)      │          │ (PM2)    │
    │ Port 3000  │          │ Port 3000│
    │ (Instance) │          │ (Cluster)│
    └────┬───────┘          └────┬─────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ↓
            ┌─────────────────┐
            │    MongoDB      │
            │   Database      │
            └─────────────────┘
```

---

## Pre-Deployment Checklist

Before deploying to Lightsail:

- [ ] AWS Lightsail account created
- [ ] Instance size: 1GB RAM, 1vCPU minimum (2GB/2vCPU recommended)
- [ ] Operating System: Ubuntu 20.04 LTS or newer
- [ ] SSH key pair generated and downloaded
- [ ] Security groups configured (ports 22, 80, 443 open)
- [ ] Domain name purchased (optional but recommended)
- [ ] Email SMTP credentials ready (for notifications)
- [ ] Database backup strategy planned
- [ ] SSL certificate ready (Let's Encrypt recommended)
- [ ] Git repository access configured

---

## AWS Lightsail Setup

### Step 1: Create Lightsail Instance

1. Log in to AWS Lightsail Console
2. Click "Create instance"
3. Select Region (closest to users)
4. Select "Ubuntu 20.04 LTS" or newer
5. Choose instance plan:
   - **Development**: 1GB RAM, 1vCPU ($5/month)
   - **Production**: 2GB RAM, 2vCPU ($10/month) ← Recommended
6. Name: `ilead-ams-prod`
7. Click "Create Instance"

### Step 2: Wait for Instance to Start

- Instance will be ready in 2-3 minutes
- Copy the public IP address
- Download SSH key pair if not already done

### Step 3: Connect to Instance

```bash
# On your local machine
ssh -i /path/to/key.pem ubuntu@<public-ip>

# Update system packages
sudo apt update && sudo apt upgrade -y
```

### Step 4: Configure Security Groups

In Lightsail Console:
1. Click on instance name
2. Go to "Networking" tab
3. Add inbound rules:
   - **Port 22** (SSH) - from your IP only
   - **Port 80** (HTTP) - from anywhere
   - **Port 443** (HTTPS) - from anywhere
   - **Port 3000** (Optional backend) - from application only
   - **Port 27017** (MongoDB) - from localhost only

---

## Backend Installation

### Step 1: Install Node.js and npm

```bash
# Install Node.js 18 LTS
curl -sL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installation
node -v  # Should show v18.x.x
npm -v   # Should show 9.x.x
```

### Step 2: Install MongoDB

```bash
# Add MongoDB repository
curl -fsSL https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse"

# Install MongoDB
sudo apt install -y mongodb-org

# Start MongoDB service
sudo systemctl start mongod
sudo systemctl enable mongod

# Verify MongoDB is running
sudo systemctl status mongod
```

### Step 3: Install PM2 (Process Manager)

```bash
sudo npm install -g pm2

# Enable PM2 to start on reboot
pm2 startup
pm2 save
```

### Step 4: Clone and Deploy Backend

```bash
# Create app directory
mkdir -p /opt/apps
cd /opt/apps

# Clone repository
git clone https://github.com/your-org/ilead-ams.git
cd ilead-ams/backend

# Install dependencies
npm install

# Create .env file (see Environment Variables section)
nano .env

# Build TypeScript (if applicable)
npm run build

# Start with PM2
pm2 start npm --name "ilead-backend" -- start

# Check status
pm2 status
```

### Step 5: Verify Backend API

```bash
# Test API endpoint
curl http://localhost:3000/api/health

# Should return: {"status": "ok"}
```

---

## Frontend Installation

### Step 1: Install Nginx

```bash
sudo apt install -y nginx

# Enable Nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

### Step 2: Deploy Frontend Build

```bash
# Create web directory
sudo mkdir -p /var/www/ilead-ams
cd /var/www/ilead-ams

# Copy frontend build files
# (Assuming dist folder from npm run build)
sudo cp -r /home/user/-AI_Powered_Predictive_CRM_Analytics_System/ilead-ams/frontend/dist/* .

# Set permissions
sudo chown -R www-data:www-data /var/www/ilead-ams
sudo chmod -R 755 /var/www/ilead-ams
```

### Step 3: Configure Nginx

Create `/etc/nginx/sites-available/ilead-ams`:

```nginx
server {
    listen 80;
    server_name _;  # Change to your domain

    root /var/www/ilead-ams;
    index index.html;

    # Serve static files with caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }

    # SPA routing - serve index.html for all routes
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API proxy to backend
    location /api/ {
        proxy_pass http://localhost:3000/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
}
```

Enable the site:

```bash
# Create symlink
sudo ln -s /etc/nginx/sites-available/ilead-ams /etc/nginx/sites-enabled/

# Remove default site
sudo rm /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

---

## Database Configuration

### Step 1: Create MongoDB Databases

```bash
# Connect to MongoDB
mongosh

# Create application database
use ilead_ams_db

# Create users collection with index
db.users.createIndex({ email: 1 }, { unique: true })

# Create apprentices collection
db.apprentices.createIndex({ userId: 1 })

# Create hour logs collection
db.hourLogs.createIndex({ apprenticeId: 1 })
db.hourLogs.createIndex({ createdAt: 1 })

# Create analytics collection
db.analytics.createIndex({ apprenticeId: 1 })

# Exit MongoDB shell
exit
```

### Step 2: Create MongoDB User

```bash
mongosh

# Switch to admin database
use admin

# Create admin user
db.createUser({
  user: "ilead_admin",
  pwd: "YOUR_SECURE_PASSWORD",
  roles: ["userAdminAnyDatabase", "dbOwner"]
})

# Create application user
db.createUser({
  user: "ilead_app",
  pwd: "YOUR_APP_PASSWORD",
  roles: [{ role: "dbOwner", db: "ilead_ams_db" }]
})

exit
```

### Step 3: Enable MongoDB Authentication

Edit `/etc/mongod.conf`:

```yaml
security:
  authorization: enabled
```

Restart MongoDB:

```bash
sudo systemctl restart mongod
```

---

## Environment Variables

### Backend (.env)

Create `/opt/apps/ilead-ams/backend/.env`:

```env
# Node Environment
NODE_ENV=production

# Server Configuration
PORT=3000
HOST=localhost

# Database Configuration
MONGODB_URI=mongodb://ilead_app:YOUR_APP_PASSWORD@localhost:27017/ilead_ams_db

# JWT Configuration
JWT_SECRET=your-super-secret-jwt-key-change-this-to-random-string
JWT_EXPIRY=7d
JWT_REFRESH_EXPIRY=30d

# Frontend URL (for CORS)
FRONTEND_URL=https://your-domain.com
# FRONTEND_URL=http://localhost:3001 (for local development)

# Email Configuration (for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
SMTP_FROM=noreply@your-domain.com

# API Documentation
API_DOCS_ENABLED=true
API_DOCS_PATH=/api/docs

# Logging
LOG_LEVEL=info
LOG_FORMAT=json

# Rate Limiting
RATE_LIMIT_WINDOW=15m
RATE_LIMIT_MAX_REQUESTS=100
```

### Frontend (.env)

Create `/var/www/ilead-ams/.env.production`:

```env
VITE_API_URL=https://your-domain.com
VITE_API_TIMEOUT=30000
VITE_APP_NAME=I-LEAD AMS
VITE_APP_VERSION=4.0.0
```

---

## SSL/TLS Configuration

### Using Let's Encrypt with Certbot

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain certificate (replace with your domain)
sudo certbot certonly --nginx -d your-domain.com -d www.your-domain.com

# Update Nginx configuration with SSL
sudo nano /etc/nginx/sites-available/ilead-ams
```

Add to Nginx config:

```nginx
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # ... rest of configuration
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

Auto-renew certificates:

```bash
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
```

---

## Testing & Verification

### Step 1: Health Checks

```bash
# Backend API health
curl https://your-domain.com/api/health

# Frontend accessibility
curl https://your-domain.com

# Check server status
pm2 status
systemctl status nginx
systemctl status mongod
```

### Step 2: Functional Testing

1. **Access Application**: Visit https://your-domain.com
2. **Login**: Use test credentials
3. **Test Analytics**:
   - Navigate to Analytics dashboard
   - Verify charts load and render
4. **Test Filtering**:
   - Go to Hour Logs
   - Apply advanced filters
   - Verify data updates
5. **Test Exports**:
   - Click export buttons
   - Verify CSV files download
6. **Test Responsive Design**:
   - Test on mobile (using browser DevTools)
   - Verify layout and functionality

### Step 3: Performance Testing

```bash
# Check CPU and memory usage
top -bn1 | head -20

# Check disk space
df -h

# Check MongoDB performance
mongosh
db.stats()
```

### Step 4: Log Monitoring

```bash
# Backend logs
pm2 logs ilead-backend

# Nginx error logs
sudo tail -f /var/log/nginx/error.log

# Nginx access logs
sudo tail -f /var/log/nginx/access.log

# MongoDB logs
sudo tail -f /var/log/mongodb/mongod.log
```

---

## Troubleshooting

### Issue: Backend not connecting to MongoDB

**Solution**:
```bash
# Check MongoDB is running
sudo systemctl status mongod

# Check MongoDB is listening on port 27017
sudo netstat -tulpn | grep 27017

# Check backend logs
pm2 logs ilead-backend

# Verify connection string in .env
```

### Issue: Frontend showing "API connection error"

**Solution**:
```bash
# Check Nginx proxy configuration
sudo nginx -t

# Check backend is running
curl http://localhost:3000/api/health

# Check browser console for CORS errors
# Verify FRONTEND_URL in backend .env
```

### Issue: SSL certificate errors

**Solution**:
```bash
# Check certificate validity
sudo certbot certificates

# Renew certificate manually
sudo certbot renew --dry-run

# Check Nginx SSL configuration
sudo nginx -t
```

### Issue: High memory usage

**Solution**:
```bash
# Restart backend
pm2 restart ilead-backend

# Check for memory leaks
pm2 monit

# Upgrade instance if needed
# Reduce log verbosity: NODE_ENV=production, LOG_LEVEL=warn
```

### Issue: Slow performance

**Solution**:
```bash
# Check database indexes
mongosh
use ilead_ams_db
db.hourLogs.getIndexes()

# Check PM2 status
pm2 status

# Upgrade instance size if CPU/memory at limit
# Enable response caching in Nginx
# Optimize MongoDB queries
```

---

## Support & Maintenance

### Regular Maintenance Tasks

**Daily**:
- Monitor server logs for errors
- Check disk space (maintain >20% free)
- Verify services are running

**Weekly**:
- Review application logs
- Check database backup status
- Monitor system performance

**Monthly**:
- Update system packages: `sudo apt update && sudo apt upgrade`
- Verify SSL certificate expiry
- Review security logs
- Backup database

**Quarterly**:
- Update Node.js dependencies
- Review and update MongoDB indexes
- Capacity planning review

### Backup Strategy

**Database Backup**:

```bash
# Create backup script: /opt/backups/backup-mongodb.sh
#!/bin/bash
BACKUP_DIR="/opt/backups/mongodb"
DATE=$(date +%Y%m%d_%H%M%S)
mongodump --uri "mongodb://ilead_app:PASSWORD@localhost:27017/ilead_ams_db" \
          --out "$BACKUP_DIR/backup_$DATE"
# Keep only last 30 days
find "$BACKUP_DIR" -type d -mtime +30 -exec rm -rf {} \;
```

Schedule with cron:

```bash
sudo crontab -e

# Add: 0 2 * * * /opt/backups/backup-mongodb.sh
```

**File Backup**:

```bash
# Backup application and configuration
tar -czf /opt/backups/ilead-ams_$(date +%Y%m%d).tar.gz \
         /opt/apps/ilead-ams \
         /etc/nginx/sites-available/ilead-ams
```

### Disaster Recovery

1. **Restore MongoDB**:
   ```bash
   mongorestore --uri "mongodb://ilead_app:PASSWORD@localhost:27017" \
                /opt/backups/mongodb/backup_latest
   ```

2. **Restore Backend**:
   ```bash
   cd /opt/apps
   rm -rf ilead-ams
   tar -xzf /opt/backups/ilead-ams_latest.tar.gz
   cd ilead-ams/backend
   npm install
   pm2 start npm --name "ilead-backend" -- start
   ```

3. **Restore Frontend**:
   ```bash
   sudo rm -rf /var/www/ilead-ams/*
   # Rebuild and redeploy frontend
   ```

---

## Monitoring & Alerts

### Install Monitoring Tools

```bash
# Install htop for system monitoring
sudo apt install -y htop

# View system stats
htop

# Install PM2 monitoring
pm2 web
# Access at http://localhost:9615
```

### Set Up Log Aggregation (Optional)

```bash
# Install Logrotate for log management
sudo apt install -y logrotate

# Nginx logs rotation
cat > /etc/logrotate.d/nginx
/var/log/nginx/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 www-data adm
    sharedscripts
    prerotate {
        if [ -d /etc/logrotate.d/httpd-prerotate.d ]; then \
            run-parts /etc/logrotate.d/httpd-prerotate.d; \
        fi
    }
    postrotate {
        [ ! -f /var/run/nginx.pid ] || kill -USR1 `cat /var/run/nginx.pid`
    }
}
```

---

## Production Checklist

Before going live:

- [ ] SSL certificate installed and verified
- [ ] Database backed up
- [ ] Environment variables configured
- [ ] All services (Backend, Frontend, MongoDB) running
- [ ] Health checks passing
- [ ] Analytics dashboard working
- [ ] Filters and exports functional
- [ ] Mobile responsiveness verified
- [ ] Error logging configured
- [ ] Database authentication enabled
- [ ] Nginx security headers set
- [ ] Rate limiting configured
- [ ] Monitoring tools installed
- [ ] Backup strategy tested
- [ ] DNS records updated (if using domain)
- [ ] Email notifications tested
- [ ] User accounts created for testing
- [ ] Firewall rules verified

---

## Quick Start Command Reference

```bash
# View all running services
pm2 status
pm2 logs

# Restart all services
pm2 restart all

# Restart Nginx
sudo systemctl restart nginx

# Check MongoDB
sudo systemctl status mongod

# Monitor system
htop

# View Nginx errors
sudo tail -f /var/log/nginx/error.log

# SSH into instance
ssh -i key.pem ubuntu@<public-ip>

# SCP files to instance
scp -i key.pem -r local/path ubuntu@<public-ip>:/remote/path
```

---

## System Architecture Files

### File Structure on Lightsail

```
/opt/apps/
└── ilead-ams/
    ├── backend/
    │   ├── src/
    │   ├── dist/
    │   ├── package.json
    │   └── .env
    └── frontend/
        ├── src/
        ├── dist/
        └── package.json

/var/www/
└── ilead-ams/
    ├── index.html
    ├── assets/
    │   ├── index-*.js
    │   └── index-*.css
    └── ...

/etc/nginx/
├── sites-available/
│   └── ilead-ams
└── sites-enabled/
    └── ilead-ams

/opt/backups/
├── mongodb/
└── ilead-ams_*.tar.gz
```

---

## Contact & Support

**Project Repository**: https://github.com/your-org/ilead-ams
**Issue Tracker**: https://github.com/your-org/ilead-ams/issues
**Documentation**: https://docs.your-domain.com
**Email Support**: support@your-domain.com

---

## Version History

| Version | Release Date | Features | Status |
|---------|-------------|----------|--------|
| V4.0 | March 2026 | Analytics, Filtering, Export | ✅ Current |
| V3.0 | Feb 2026 | Approvals, User Management | ✅ Stable |
| V2.0 | Jan 2026 | Hour Logging, Apprentice Mgmt | ✅ Legacy |
| V1.0 | Dec 2025 | Auth, Dashboard | ✅ Legacy |

---

**Document Version**: 1.0
**Last Updated**: March 17, 2026
**Next Review**: April 2026

