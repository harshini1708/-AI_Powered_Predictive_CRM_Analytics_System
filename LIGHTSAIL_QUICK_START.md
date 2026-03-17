# I-LEAD AMS - Lightsail Quick Start Guide
## 5-Minute Setup for AWS Lightsail

**For detailed instructions, see `DEPLOYMENT_GUIDE.md`**

---

## Prerequisites
- AWS Lightsail instance (Ubuntu 20.04+ or newer) with 1GB+ RAM
- SSH access to instance
- Domain name (optional but recommended)

---

## Step 1: SSH Into Your Instance (2 minutes)

```bash
# Replace with your instance IP
ssh -i your-key.pem ubuntu@12.34.56.78

# Update system
sudo apt update && sudo apt upgrade -y
```

---

## Step 2: Install Core Dependencies (3 minutes)

```bash
# Install Node.js 18
curl -sL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install MongoDB
curl -fsSL https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse"
sudo apt install -y mongodb-org
sudo systemctl start mongod && sudo systemctl enable mongod

# Install PM2 and Nginx
sudo npm install -g pm2
sudo apt install -y nginx

# Enable PM2 startup
pm2 startup
pm2 save
```

---

## Step 3: Deploy Backend (2 minutes)

```bash
# Clone repository
cd /opt/apps 2>/dev/null || mkdir -p /opt/apps && cd /opt/apps
git clone https://github.com/your-org/ilead-ams.git
cd ilead-ams/backend

# Install and configure
npm install

# Create .env file with your settings
cat > .env << EOF
NODE_ENV=production
PORT=3000
MONGODB_URI=mongodb://localhost:27017/ilead_ams_db
JWT_SECRET=your-random-secret-key-here-change-this
FRONTEND_URL=https://your-domain.com
EOF

# Start backend
npm run build 2>/dev/null || true
pm2 start npm --name "ilead-backend" -- start
pm2 save

# Verify
curl http://localhost:3000/api/health
```

---

## Step 4: Deploy Frontend (1 minute)

```bash
# Create web directory
sudo mkdir -p /var/www/ilead-ams
cd ilead-ams/frontend

# Build
npm install
npm run build

# Copy to web directory
sudo cp -r dist/* /var/www/ilead-ams/
sudo chown -R www-data:www-data /var/www/ilead-ams
```

---

## Step 5: Configure Nginx (1 minute)

```bash
# Create Nginx config
sudo tee /etc/nginx/sites-available/ilead-ams > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;

    root /var/www/ilead-ams;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://localhost:3000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
EOF

# Enable and restart
sudo ln -sf /etc/nginx/sites-available/ilead-ams /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx
```

---

## Step 6: Verify Everything Works

```bash
# Check services
pm2 status
sudo systemctl status nginx
sudo systemctl status mongod

# Test endpoints
curl http://localhost/           # Frontend
curl http://localhost/api/health # Backend API

echo "✅ Setup Complete! Access at: http://your-instance-ip"
```

---

## Add SSL Certificate (Optional but Recommended)

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get certificate (replace with your domain)
sudo certbot certonly --nginx -d your-domain.com -d www.your-domain.com

# Update Nginx config with SSL and redirect HTTP to HTTPS
# See DEPLOYMENT_GUIDE.md for complete SSL configuration
```

---

## Useful Commands

```bash
# View backend logs
pm2 logs ilead-backend

# Restart backend
pm2 restart ilead-backend

# Restart Nginx
sudo systemctl restart nginx

# Monitor system
htop

# Check database
mongosh

# Stop services
pm2 stop ilead-backend
sudo systemctl stop nginx
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 3000 in use | `sudo lsof -i :3000` then kill the process |
| Nginx won't start | `sudo nginx -t` to check config |
| MongoDB not running | `sudo systemctl restart mongod` |
| API connection error | Check `FRONTEND_URL` in backend .env |
| Frontend not loading | Check `/var/log/nginx/error.log` |

---

## Next Steps

1. **Update environment variables** in backend `.env` for your domain
2. **Configure SSL certificate** for HTTPS
3. **Set up database backups** (see DEPLOYMENT_GUIDE.md)
4. **Enable email notifications** (configure SMTP in .env)
5. **Monitor performance** with `htop` and `pm2 monit`
6. **Update DNS records** to point to your Lightsail instance IP

---

## Full Documentation

For complete deployment instructions including:
- Database configuration
- SSL/TLS setup
- Monitoring and alerts
- Backup strategy
- Troubleshooting guide
- Production checklist

**See: `DEPLOYMENT_GUIDE.md`**

---

**System Version**: V4 (Phase 7C Complete)
**Ready for Production**: ✅ Yes
**Support**: See DEPLOYMENT_GUIDE.md for contact info

