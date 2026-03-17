# Production Deployment Checklist

**Status**: Ready for Deployment ✅
**Date**: March 17, 2026
**Environment**: Production

---

## 🔧 PRE-DEPLOYMENT SETUP

### Server Preparation
- [ ] **Provision server** (Ubuntu 20.04 LTS, 4GB+ RAM, 80GB+ SSD)
- [ ] **Set static IP address** on server
- [ ] **Update system packages**: `sudo apt update && sudo apt upgrade -y`
- [ ] **Install Docker & Docker Compose**
  ```bash
  curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
  ```
- [ ] **Install required tools**: `sudo apt install -y git curl wget unzip`
- [ ] **Configure firewall** (UFW)
  ```bash
  sudo ufw enable
  sudo ufw allow 22/tcp    # SSH
  sudo ufw allow 80/tcp    # HTTP
  sudo ufw allow 443/tcp   # HTTPS
  ```

### Domain & SSL Setup
- [ ] **Purchase domain name** (or use existing)
- [ ] **Point DNS records** to server IP
  - A record: `your-domain.com` → server IP
  - CNAME: `www.your-domain.com` → `your-domain.com`
- [ ] **Generate SSL certificates** with Let's Encrypt
  ```bash
  sudo apt install -y certbot
  sudo certbot certonly --standalone -d your-domain.com -d api.your-domain.com
  ```
- [ ] **Copy certificates to project**
  ```bash
  sudo mkdir -p /etc/ilead-ams/ssl
  sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem /etc/ilead-ams/ssl/cert.pem
  sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem /etc/ilead-ams/ssl/key.pem
  ```

### Repository Setup
- [ ] **Clone repository**
  ```bash
  git clone https://github.com/your-org/ilead-ams.git
  cd ilead-ams
  ```
- [ ] **Checkout correct branch**
  ```bash
  git checkout claude/migrate-nodejs-backend-ISFDE
  ```

---

## 🔐 ENVIRONMENT CONFIGURATION

### Backend Environment
- [ ] **Create .env file from template**
  ```bash
  cp ilead-ams/backend/.env.example ilead-ams/backend/.env
  ```
- [ ] **Set environment variables** (edit `.env`):
  ```
  NODE_ENV=production
  PORT=3001
  MONGODB_URI=mongodb://user:password@mongodb:27017/ilead_ams?authSource=admin
  JWT_SECRET=<generate: openssl rand -hex 32>
  FRONTEND_URL=https://your-domain.com
  CORS_ORIGIN=https://your-domain.com
  
  # Email Configuration
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587
  SMTP_USER=your-email@gmail.com
  SMTP_PASS=<app-specific-password>
  SMTP_FROM=noreply@your-domain.com
  
  # Optional: AWS S3
  AWS_REGION=us-east-1
  AWS_ACCESS_KEY_ID=<your-access-key>
  AWS_SECRET_ACCESS_KEY=<your-secret-key>
  AWS_S3_BUCKET=ilead-ams-uploads
  ```

### Frontend Environment
- [ ] **Create .env.local file from template**
  ```bash
  cp ilead-ams/frontend/.env.example ilead-ams/frontend/.env.local
  ```
- [ ] **Set environment variables**:
  ```
  REACT_APP_API_URL=https://api.your-domain.com
  REACT_APP_ENV=production
  REACT_APP_ENABLE_ANALYTICS=true
  ```

### Docker Compose Secrets
- [ ] **Update docker-compose.yml** with actual credentials:
  ```yaml
  environment:
    MONGO_USER: <secure-username>
    MONGO_PASSWORD: <generate: openssl rand -hex 32>
    MONGO_DB: ilead_ams_prod
  ```

---

## 🚀 DEPLOYMENT EXECUTION

### Pre-deployment Validation
- [ ] **Run validation script**
  ```bash
  ./validate-build.sh
  # Should show: ✅ ALL VALIDATIONS PASSED
  ```

### Docker Image Building
- [ ] **Build Docker images**
  ```bash
  docker-compose build
  ```
- [ ] **Verify images created**
  ```bash
  docker images | grep ilead
  ```

### Starting Services
- [ ] **Start all services**
  ```bash
  ./scripts/deploy.sh up
  ```
- [ ] **Wait for services to be healthy** (2-3 minutes)
  ```bash
  docker-compose ps
  # All should show "healthy" or "running"
  ```

### Database Initialization
- [ ] **Run database migrations** (if needed)
  ```bash
  docker-compose exec backend npm run migrate
  ```
- [ ] **Seed initial data** (if needed)
  ```bash
  docker-compose exec backend npm run seed
  ```

### Health Verification
- [ ] **Check system health**
  ```bash
  ./scripts/health-check.sh
  # Should show: ✅ All systems operational
  ```

### Domain Verification
- [ ] **Test frontend accessibility**
  ```bash
  curl -i https://your-domain.com
  # Should return HTTP 200
  ```
- [ ] **Test API accessibility**
  ```bash
  curl -i https://your-domain.com/api/health
  # Should return API health status
  ```

---

## 📊 POST-DEPLOYMENT CHECKS

### Functionality Testing
- [ ] **Login with test user account**
  - Navigate to https://your-domain.com
  - Test login with admin credentials
  - Verify dashboard loads correctly

- [ ] **Test email notifications**
  - Trigger a notification-generating action
  - Verify email received within 5 minutes
  - Check email formatting and links

- [ ] **Test reporting system**
  - Create a test report
  - Verify data generation works
  - Test export to CSV/Excel/PDF

- [ ] **Test dark mode**
  - Toggle dark/light mode
  - Verify theme switches correctly
  - Check all components render properly

- [ ] **Test responsiveness**
  - View site on mobile device
  - Verify layout adapts correctly
  - Test touch interactions

### Database Verification
- [ ] **Verify MongoDB is running**
  ```bash
  docker-compose exec mongodb mongosh -u admin -p
  > use ilead_ams_prod
  > db.users.countDocuments()
  ```

- [ ] **Check database backups**
  ```bash
  ls -lh ./backups/
  # Should show recent backup file
  ```

### Performance Monitoring
- [ ] **Monitor resource usage**
  ```bash
  docker stats
  # Check CPU, Memory usage is reasonable
  ```

- [ ] **Check response times**
  - API endpoint: Should be < 200ms
  - Frontend page load: Should be < 3s
  - Database queries: Should be < 100ms

### Security Verification
- [ ] **Verify HTTPS is enforced**
  ```bash
  curl -i http://your-domain.com
  # Should redirect to HTTPS (301)
  ```

- [ ] **Check security headers**
  ```bash
  curl -i https://your-domain.com | grep -i "Strict-Transport"
  # Should show HSTS header
  ```

- [ ] **Verify rate limiting**
  - Make 100+ requests to API
  - Verify 429 response after limit exceeded

---

## 🔄 ONGOING OPERATIONS

### Daily
- [ ] **Review health checks**
  ```bash
  ./scripts/health-check.sh
  ```
- [ ] **Check error logs**
  ```bash
  docker-compose logs backend | tail -50
  ```
- [ ] **Monitor user activity** (check analytics if enabled)

### Weekly
- [ ] **Verify backups are being created**
  ```bash
  ls -lh ./backups/ | head -5
  ```
- [ ] **Review system metrics** (CPU, Memory, Disk)
- [ ] **Test backup restoration** (on staging, not production)
- [ ] **Review and update dependencies** (if patches available)

### Monthly
- [ ] **Security updates**: Check and apply security patches
- [ ] **Performance analysis**: Review metrics and optimize if needed
- [ ] **Disaster recovery drill**: Test full backup restore
- [ ] **SSL certificate renewal**: Let's Encrypt auto-renewal should handle this

---

## 🆘 TROUBLESHOOTING

### If services won't start:
```bash
# Check logs
docker-compose logs

# Verify environment variables are set
cat ilead-ams/backend/.env | head -20

# Rebuild images
docker-compose build --no-cache

# Restart services
./scripts/deploy.sh down
./scripts/deploy.sh up
```

### If database won't connect:
```bash
# Check MongoDB is running
docker-compose ps mongodb

# Verify credentials in .env match docker-compose.yml
# Check firewall isn't blocking port 27017

# Restart MongoDB
docker-compose restart mongodb
```

### If website shows blank page:
```bash
# Check frontend logs
docker-compose logs frontend

# Check browser console for errors (F12)
# Verify API_URL is correct in frontend .env

# Check Nginx configuration
docker-compose exec nginx nginx -t
```

### If emails aren't sending:
```bash
# Check SMTP credentials are correct
# Verify SMTP_USER has "Less secure apps" enabled (Gmail)
# Check spam folder for emails
# Review backend logs: docker-compose logs backend | grep -i email
```

---

## 📋 FINAL SIGN-OFF

- [ ] **All checks passed**
- [ ] **Team notified of deployment**
- [ ] **Backup verified**
- [ ] **Monitoring configured**
- [ ] **On-call support assigned**
- [ ] **Documentation updated**
- [ ] **Deployment complete and verified** ✅

---

## 🚨 ROLLBACK PROCEDURE

If critical issues occur:

```bash
# 1. Stop current services
./scripts/deploy.sh down

# 2. Restore database from backup
./scripts/restore-database.sh ./backups/latest.archive

# 3. Checkout previous version
git checkout <previous-commit-hash>

# 4. Restart services
./scripts/deploy.sh up

# 5. Verify
./scripts/health-check.sh
```

---

**Deployment Status**: Ready ✅
**Last Updated**: March 17, 2026
**Next Review**: After deployment

