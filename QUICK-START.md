# I-LEAD AMS - Quick Start Guide

## Get Everything Running in 5 Minutes

### Prerequisites
- Docker & Docker Compose (recommended)
- OR: Node.js 18+, PostgreSQL 15+

### Option 1: Docker (Recommended - Fastest)

#### Terminal 1: Start Backend API
```bash
cd ilead-ams

# Initialize and start backend
./docker-init.sh development

# Wait for output: "I-LEAD AMS Backend Server" with port 3000
```

#### Terminal 2: Start Frontend
```bash
cd ilead-ams/frontend

# Install dependencies (first time only)
npm install

# Start frontend dev server
npm run dev

# Wait for output: "VITE v4.4.9 ready in XXX ms"
```

#### Terminal 3 (Optional): Watch logs
```bash
cd ilead-ams
docker-compose logs -f api
```

### Access the Application

**Frontend**: http://localhost:3001
**Backend API Docs**: http://localhost:3000/docs

---

## Test Accounts

### Super Admin
- Email: `hector.torres@i-leadusa.org`
- PIN: `071676` (when prompted)

### Administrator
- Email: `admin@i-leadusa.org`

### Creating Test Accounts
1. Click "Create Account" on login page
2. Fill in details
3. Select role: apprentice, supervisor, journeyworker, or administrator
4. Account will be pending approval (admins can approve)

---

## Architecture

```
http://localhost:3001 (Frontend)
         ↓
http://localhost:3000 (Backend API)
         ↓
PostgreSQL Database (Port 5432)
```

---

## Key Features Working

✅ User Registration & Login
✅ Role-Based Navigation (Sidebar adjusts per role)
✅ Dashboard with Real-Time Statistics
✅ Submit Apprenticeship Hours (OJT/RTI)
✅ View Hour Log List
✅ View Apprentices
✅ API Documentation (Swagger UI at /docs)

---

## Common Issues

### Port Already in Use

```bash
# Change port for frontend
API_PORT=3001 npm run dev

# Change backend port
cd ilead-ams
API_PORT=3001 docker-compose up
```

### Database Connection Error

```bash
# Check if database is running
cd ilead-ams
docker-compose ps

# If not running:
docker-compose up -d db

# Wait 5 seconds for database to initialize
```

### Frontend Can't Connect to API

1. Check backend is running: `http://localhost:3000/health`
2. Ensure CORS is enabled in backend
3. Check VITE_API_URL in frontend/.env.local matches backend URL

### Clear Cache & Reinstall

```bash
# Frontend
cd ilead-ams/frontend
rm -rf node_modules package-lock.json
npm install
npm run dev

# Backend
cd ilead-ams
docker-compose down -v
./docker-init.sh development
```

---

## Next Steps

### For Development
1. All code changes auto-reload (Vite hot module replacement)
2. Backend changes require container rebuild: `docker-compose restart api`
3. Check API documentation at `http://localhost:3000/docs`

### Testing the Full Flow

1. **Register a new user**
   - Click "Create Account"
   - Fill in details as an apprentice
   - Note: Account may need admin approval

2. **Submit hours**
   - Login as apprentice
   - Navigate to "Submit Hours"
   - Fill in OJT domain, hours, and description
   - Click "Submit Hours for Approval"

3. **Approve hours** (as supervisor/admin)
   - Login as supervisor or administrator
   - Navigate to "Hour Approvals"
   - Review and approve submitted hours

4. **View progress**
   - Login as apprentice
   - Check dashboard for updated statistics

---

## Useful Commands

### Backend (in `ilead-ams/` directory)
```bash
# Start services
docker-compose up

# Stop services
docker-compose down

# View logs
docker-compose logs -f api

# Run tests
docker-compose exec api npm test

# Access database shell
docker-compose exec db psql -U postgres

# Run migrations
docker-compose exec api npm run migrate
```

### Frontend (in `ilead-ams/frontend/` directory)
```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint

# Format code
npm run format
```

---

## Stopping Everything

```bash
# Terminal 1 (Backend): Press Ctrl+C
# Terminal 2 (Frontend): Press Ctrl+C

# Stop Docker containers
cd ilead-ams
docker-compose down

# Optional: Remove all data and start fresh
docker-compose down -v
```

---

## What's Working

### Phase 7 Progress
- ✅ Phase 7A: Complete API integration layer
- ✅ Phase 7B (Milestone 1): Authentication & Navigation
- ✅ Phase 7B (Milestone 2): Dashboard Statistics
- ✅ Phase 7B (Milestone 3): Routing & Core Pages
- ⏳ Phase 7B (Milestone 4): Full Component Completion
- ⏳ Phase 7B (Milestone 5): Testing & Optimization

### API Endpoints Available
- Authentication (register, login, refresh)
- Apprentices (list, get, create, update)
- Hour Logs (submit, list, approve, reject)
- Users (list, get, create, update, approve)
- Programs (list, get)
- Analytics (dashboard, domain progress, heatmap)
- Documents (upload, download)

---

## Environment Configuration

### Frontend (.env.local)
```
VITE_API_URL=http://localhost:3000
VITE_API_TIMEOUT=30000
```

### Backend (.env.docker)
Pre-configured for Docker - no changes needed for local development

---

## API Documentation

Once running, view API docs at:
- **Swagger UI**: http://localhost:3000/docs
- **OpenAPI JSON**: http://localhost:3000/openapi.json

Try endpoints directly from Swagger UI:
1. Click on an endpoint
2. Click "Try it out"
3. Fill in parameters
4. Click "Execute"

---

## Need Help?

1. Check application logs in terminal
2. View API docs at http://localhost:3000/docs
3. Check browser console (F12) for JavaScript errors
4. Review error messages in alert boxes on screen

---

## Production Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for production setup instructions.

---

**Version**: 1.0.0
**Status**: 🚀 Ready to Use
**Last Updated**: March 17, 2026
