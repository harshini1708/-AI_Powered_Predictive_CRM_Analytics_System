# I-LEAD AMS Migration - Quick Start Guide

## For Hector Torres & Development Team

### What's Been Set Up

1. **Complete Backend Scaffolding**
   - Express.js server with all middleware
   - PostgreSQL database schema (migrations)
   - JWT authentication structure
   - Role-based access control
   - All API route stubs

2. **Documentation**
   - Project brief with full specifications
   - 10-phase migration plan
   - Implementation status tracker
   - Backend README with API docs

3. **Tech Stack**
   - Express.js + PostgreSQL
   - Knex.js for migrations
   - JWT + bcrypt for auth
   - Joi for validation

### Get Started Locally

```bash
cd ilead-ams/backend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Edit .env with your local PostgreSQL credentials
# DB_HOST=localhost
# DB_USER=postgres
# DB_PASSWORD=<your-password>
# DB_NAME=ilead_ams

# Create database
createdb ilead_ams

# Run migrations
npm run migrate

# Start dev server (with auto-reload)
npm run dev
```

Server runs at `http://localhost:3000`

Test endpoint: `curl http://localhost:3000/health`

### Next Steps (Priority Order)

#### 1. Implement Authentication (2-3 hours)
- [ ] `src/controllers/authController.js` - register, login, PIN verification
- [ ] Password hashing with bcrypt
- [ ] JWT token generation
- [ ] Test with curl or Postman

#### 2. Create Seed Data (1 hour)
- [ ] `src/seeds/001_initial_data.js`
- [ ] Default programs (RBT-REG, ELEC-REG, SM-REG)
- [ ] Test users: super_admin, admin, supervisor, apprentice
- [ ] Run: `npm run seed`

#### 3. Implement User Management (1-2 hours)
- [ ] User CRUD controllers
- [ ] Account approval workflow
- [ ] Role-based filtering

#### 4. Implement Apprentice Management (2-3 hours)
- [ ] Apprentice CRUD
- [ ] Progress dashboard calculations
- [ ] Competency map/heat map

#### 5. Implement Hour Logs & Approval (3-4 hours)
- [ ] Hour submission
- [ ] Rubric scoring
- [ ] Domain tracking

### File Structure Reference

```
ilead-ams/
├── ILEAD-AMS-PROJECT-BRIEF.md    ← Read for full specifications
├── MIGRATION-PLAN.md              ← Read for architecture decisions
├── IMPLEMENTATION-STATUS.md       ← Track progress here
├── I-LEAD-AMS-V4.html            ← Current production code (reference)
└── backend/
    ├── src/
    │   ├── server.js             ← Express app (ready to run)
    │   ├── routes/               ← API endpoints (stubs only)
    │   ├── controllers/          ← 👈 START IMPLEMENTING HERE
    │   ├── models/               ← Data access layer (create as needed)
    │   ├── middleware/           ← Auth, validation, etc. (ready)
    │   └── migrations/           ← Database schema (ready)
    ├── package.json
    ├── knexfile.js
    ├── .env.example
    └── README.md                 ← Full API documentation
```

### Key Endpoints to Implement

**Auth** (POST):
- `/api/auth/register` - Create account
- `/api/auth/login` - Login, get JWT
- `/api/auth/verify-pin` - Verify super admin PIN
- `/api/auth/refresh-token` - Refresh JWT

**Users** (Admin only):
- `GET /api/users` - List
- `POST /api/users` - Create
- `PUT /api/users/:id` - Update
- `DELETE /api/users/:id` - Delete

**Apprentices**:
- `GET /api/apprentices` - List (role-filtered)
- `POST /api/apprentices` - Create (admin)
- `GET /api/apprentices/:id/progress` - Dashboard
- `GET /api/apprentices/:id/competency` - Heat map

**Hour Logs**:
- `POST /api/hour-logs` - Submit (apprentice)
- `GET /api/hour-logs?status=pending` - Pending approvals
- `PUT /api/hour-logs/:id/approve` - Approve with rubric

### Database Tables Ready

All tables created in migration. Key ones:

- **users** - Auth, roles, organizations
- **apprentices** - Program enrollment, supervisor tracking
- **hour_logs** - Submissions with rubric scoring
- **programs** - OJT/RTI targets
- **ratings** - Approved assessments

### Testing

Use Postman or curl. Example:

```bash
# Health check
curl http://localhost:3000/health

# List users (will fail until auth implemented)
curl -H "Authorization: Bearer <token>" \
     http://localhost:3000/api/users
```

### Important Notes

1. **Super Admin PIN**
   - Default test PIN: `071676`
   - Stored as bcrypt hash in database
   - Required for all super admin operations

2. **Role-Based Access**
   - Middleware checks role on protected routes
   - Supervisors see only their apprentices
   - All queries filtered by organization_id

3. **Email/SMS**
   - Nodemailer configured for emails
   - Clearstream API for SMS (needs new API key)
   - Background job queuing not yet implemented

4. **Frontend Integration**
   - React will call `/api` endpoints
   - CORS already configured
   - JWT token stored in browser localStorage

### Support

- Check `backend/README.md` for full setup docs
- Review `MIGRATION-PLAN.md` for architecture decisions
- See `ILEAD-AMS-PROJECT-BRIEF.md` for full specifications

### Deployment Later (After Phase 1)

```bash
# Build Docker image (not yet configured)
docker build -t ilead-ams-backend .

# Deploy to AWS Lightsail
scp -r backend/ ubuntu@98.94.220.9:/home/ubuntu/ilead-ams/
```

---

**Current Status:** Phase 1 - Backend Implementation Started
**Branch:** `claude/migrate-nodejs-backend-ISFDE`
**Ready to code:** Yes! Start with authentication controllers.

Good luck! 🚀
