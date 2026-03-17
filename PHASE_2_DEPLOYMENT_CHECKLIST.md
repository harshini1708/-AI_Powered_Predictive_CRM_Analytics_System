# Phase 2 Deployment Checklist

**Status**: Ready for Execution 🚀
**Estimated Time**: 1-2 hours
**Branch**: claude/migrate-nodejs-backend-ISFDE

---

## 📋 Prerequisites Checklist

Before starting deployment, verify you have:

### Infrastructure
- [ ] Server access (Ubuntu 20.04 LTS recommended)
- [ ] Server IP address or domain
- [ ] 4GB+ RAM, 80GB+ disk space
- [ ] SSH access configured
- [ ] Firewall rules for ports 22, 80, 443

### Domain & SSL
- [ ] Domain name registered or existing
- [ ] DNS access to update A records
- [ ] SSL certificate path (or Let's Encrypt setup)
- [ ] Domain pointing to server IP

### Credentials & Configuration
- [ ] MongoDB credentials (username/password)
- [ ] JWT secret (generate with: `openssl rand -hex 32`)
- [ ] Gmail account for email + app-specific password
- [ ] AWS S3 credentials (optional, for file uploads)

### Access & Permissions
- [ ] Git repository access
- [ ] SSH key for git deployment
- [ ] Docker Hub account (optional, for custom registries)

---

## 🚀 Deployment Execution Phases

### Phase 2A: Server & Environment Setup (20 minutes)
**Goal**: Server ready with Docker and all system dependencies

```bash
# Follow these steps:
1. [ ] SSH into server
2. [ ] Run system updates: sudo apt update && sudo apt upgrade -y
3. [ ] Install Docker: curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
4. [ ] Install Docker Compose
5. [ ] Setup firewall: sudo ufw enable && sudo ufw allow 22/tcp && sudo ufw allow 80/tcp && sudo ufw allow 443/tcp
6. [ ] Verify: docker --version && docker-compose --version
```

**Status**: [ ] Complete

---

### Phase 2B: Domain & SSL Setup (15 minutes)
**Goal**: Domain and SSL certificates configured

```bash
# Follow these steps:
1. [ ] Update DNS A records to point to server IP
2. [ ] Wait for DNS propagation (check with: nslookup your-domain.com)
3. [ ] Install certbot: sudo apt install -y certbot
4. [ ] Generate certificate: sudo certbot certonly --standalone -d your-domain.com
5. [ ] Copy certs: sudo mkdir -p /opt/ilead-ams/ssl && sudo cp /etc/letsencrypt/live/your-domain.com/* /opt/ilead-ams/ssl/
6. [ ] Set permissions: sudo chown -R $USER:$USER /opt/ilead-ams
```

**Status**: [ ] Complete

---

### Phase 2C: Repository & Code Setup (10 minutes)
**Goal**: Code checked out and ready for deployment

```bash
# Follow these steps:
1. [ ] Create directory: mkdir -p /opt
2. [ ] Clone repo: cd /opt && git clone https://github.com/your-org/-AI_Powered_Predictive_CRM_Analytics_System.git
3. [ ] Checkout branch: cd -AI_Powered_Predictive_CRM_Analytics_System && git checkout claude/migrate-nodejs-backend-ISFDE
4. [ ] Verify: git branch -vv (should show tracking branch)
```

**Status**: [ ] Complete

---

### Phase 2D: Configuration Files (15 minutes)
**Goal**: All environment variables configured with production values

#### Backend Configuration
```bash
# Create from template
cp ilead-ams/backend/.env.example ilead-ams/backend/.env
```

**Edit ilead-ams/backend/.env** with these values:
```
[ ] NODE_ENV=production
[ ] PORT=3001
[ ] MONGODB_URI=mongodb://admin:PASSWORD@mongodb:27017/ilead_ams?authSource=admin
[ ] JWT_SECRET=<32-character hex from: openssl rand -hex 32>
[ ] FRONTEND_URL=https://your-domain.com
[ ] CORS_ORIGIN=https://your-domain.com
[ ] SMTP_HOST=smtp.gmail.com
[ ] SMTP_PORT=587
[ ] SMTP_USER=your-email@gmail.com
[ ] SMTP_PASS=your-app-specific-password
[ ] SMTP_FROM=noreply@your-domain.com
[ ] AWS_S3_BUCKET=ilead-ams-prod (if using S3)
```

#### Frontend Configuration
```bash
# Create from template
cp ilead-ams/frontend/.env.example ilead-ams/frontend/.env.local
```

**Edit ilead-ams/frontend/.env.local** with:
```
[ ] REACT_APP_API_URL=https://api.your-domain.com
[ ] REACT_APP_ENV=production
[ ] REACT_APP_ENABLE_ANALYTICS=true
```

#### Docker Compose Secrets
```bash
# Edit docker-compose.yml
nano docker-compose.yml
```

**Update these environment variables**:
```
[ ] MONGO_INITDB_ROOT_USERNAME=admin
[ ] MONGO_INITDB_ROOT_PASSWORD=<secure-password>
[ ] MONGO_INITDB_DATABASE=ilead_ams
```

**Status**: [ ] Complete

---

### Phase 2E: Build & Deploy (30 minutes)
**Goal**: Docker images built and all services running

```bash
# Validation
1. [ ] Run: ./validate-build.sh
        Expected: ✅ ALL VALIDATIONS PASSED (46/46)

# Build
2. [ ] Run: docker-compose build
        Expected: All images built successfully
3. [ ] Verify: docker images | grep ilead

# Start services
4. [ ] Run: ./scripts/deploy.sh up
        OR: docker-compose up -d
        Expected: All containers starting
5. [ ] Wait 2-3 minutes for services to be healthy
6. [ ] Check: docker-compose ps
        Expected: All services showing "healthy" or "running"

# Initialize database
7. [ ] Run: docker-compose exec backend npm run migrate
        Expected: Migrations complete
8. [ ] (Optional) Run: docker-compose exec backend npm run seed
        Expected: Sample data loaded
```

**Status**: [ ] Complete

---

### Phase 2F: Post-Deployment Verification (20 minutes)
**Goal**: All systems operational and accessible

#### Health Checks
```bash
[ ] Run: ./scripts/health-check.sh
    Expected: ✅ All systems operational

[ ] Check: docker-compose ps
    Expected: All services "healthy" or "running"

[ ] Monitor: docker stats
    Expected: CPU < 20%, Memory < 1GB total
```

#### Accessibility Tests
```bash
[ ] Frontend: curl -I https://your-domain.com
    Expected: HTTP 200

[ ] API: curl -I https://api.your-domain.com/health
    Expected: HTTP 200

[ ] Redirect: curl -I http://your-domain.com
    Expected: HTTP 301 with Location: https://

[ ] Browser: Open https://your-domain.com
    Expected: Login page loads, responsive, no errors
```

#### Functionality Tests
```bash
[ ] Login with test user
    - Navigate to https://your-domain.com
    - Login with admin credentials
    - Dashboard loads

[ ] Email notifications
    - Trigger action that sends email
    - Check inbox within 5 minutes
    - Email formatting correct

[ ] Dark mode
    - Toggle dark/light mode
    - Colors apply correctly
    - All components visible

[ ] Mobile responsiveness
    - View on mobile device (or browser dev tools)
    - Layout adapts correctly
    - Touch interactions work
```

#### Security Verification
```bash
[ ] SSL/TLS: curl -I https://your-domain.com | grep Strict-Transport
    Expected: HSTS header present

[ ] Rate limiting: Make 100+ requests, verify 429 after limit
    Expected: Returns 429 Too Many Requests

[ ] No console errors: Open browser DevTools > Console
    Expected: No errors or warnings
```

**Status**: [ ] Complete

---

### Phase 2G: Backup & Monitoring Setup (10 minutes)
**Goal**: Automated backups and monitoring in place

```bash
# Setup daily backup
[ ] Edit crontab: crontab -e

# Add this line (3 AM UTC daily):
0 3 * * * /opt/-AI_Powered_Predictive_CRM_Analytics_System/scripts/backup-database.sh

# Test backup
[ ] Run: ./scripts/backup-database.sh
    Expected: Backup file created in ./backups/

# Test restore procedure (don't actually restore in production)
[ ] Verify restore script exists: ls ./scripts/restore-database.sh
```

**Status**: [ ] Complete

---

## 🎯 Deployment Summary

### Checklist Status
- [ ] Prerequisites verified
- [ ] Phase 2A: Server setup complete
- [ ] Phase 2B: Domain & SSL configured
- [ ] Phase 2C: Code checked out
- [ ] Phase 2D: Configuration files created
- [ ] Phase 2E: Services deployed
- [ ] Phase 2F: Verification complete
- [ ] Phase 2G: Backups configured

### Final Sign-Off
- [ ] All services operational ✅
- [ ] Domain accessible via HTTPS ✅
- [ ] Email notifications working ✅
- [ ] Database backed up daily ✅
- [ ] Health checks passing ✅
- [ ] Monitoring in place ✅

---

## 📞 Getting Help

### Common Issues

**Docker not found?**
```bash
# Make sure docker group is configured
sudo usermod -aG docker $USER
newgrp docker
# Then log out and back in
```

**Database connection fails?**
```bash
# Verify credentials in .env
# Check MongoDB logs:
docker-compose logs mongodb

# Test connection:
docker-compose exec mongodb mongosh -u admin -p
```

**Domain not resolving?**
```bash
# Check DNS propagation
nslookup your-domain.com
# May take 30 minutes to propagate

# Or use online tool: https://dnschecker.org
```

**Email not sending?**
```bash
# Verify SMTP_PASS is app-specific password, not regular Gmail password
# Gmail app-specific password setup:
# 1. Enable 2-factor authentication on Gmail
# 2. Go to myaccount.google.com/apppasswords
# 3. Generate password for "Mail" + "Windows/Mac/Linux"
# 4. Use that password as SMTP_PASS
```

### Support Resources
- **Detailed Guide**: See PHASE_2_DEPLOYMENT_GUIDE.md
- **Complete Checklist**: See PRODUCTION_DEPLOYMENT_CHECKLIST.md
- **Docker Docs**: https://docs.docker.com
- **Let's Encrypt**: https://letsencrypt.org/getting-started

---

## 🚀 Next Steps After Deployment

Once Phase 2 is complete and verified:

1. **Monitor for 24 hours** - Ensure stability
2. **Proceed to Phase 3** - Complete Phase 7D Week 3 (RBAC & Performance optimization)
3. **Then Phase 4** - Start Phase 7E mobile app development

---

**Document Version**: 1.0
**Created**: March 17, 2026
**Last Updated**: March 17, 2026
**Status**: Ready for Production Deployment ✅
