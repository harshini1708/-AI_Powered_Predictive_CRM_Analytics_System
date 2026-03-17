# I-LEAD AMS Node.js Backend Migration Plan

## Phase 1: Project Structure & Setup

### Directory Structure
```
ilead-ams/
├── backend/
│   ├── src/
│   │   ├── server.js                    # Express app entry point
│   │   ├── config/
│   │   │   ├── database.js              # Database connection & setup
│   │   │   ├── auth.js                  # JWT/auth configuration
│   │   │   └── constants.js             # App-wide constants (programs, domains, etc.)
│   │   ├── models/
│   │   │   ├── User.js
│   │   │   ├── Apprentice.js
│   │   │   ├── HourLog.js
│   │   │   ├── Program.js
│   │   │   ├── Organization.js
│   │   │   └── Document.js
│   │   ├── routes/
│   │   │   ├── auth.js                  # Login, register, password reset
│   │   │   ├── users.js                 # User CRUD (admin only)
│   │   │   ├── apprentices.js           # Apprentice CRUD & dashboard
│   │   │   ├── hourLogs.js              # Hour submission & approval
│   │   │   ├── programs.js              # Program management
│   │   │   ├── organizations.js         # Org management (super admin)
│   │   │   ├── documents.js             # File upload/storage
│   │   │   └── analytics.js             # Dashboard stats & reports
│   │   ├── middleware/
│   │   │   ├── auth.js                  # JWT verification
│   │   │   ├── roleCheck.js             # Role-based access control
│   │   │   ├── errorHandler.js          # Centralized error handling
│   │   │   └── validation.js            # Input validation
│   │   ├── utils/
│   │   │   ├── webhooks.js              # Make.com webhook handlers
│   │   │   ├── ids.js                   # ID generation (USR-, APP-, LOG-, etc.)
│   │   │   └── translations.js          # Bilingual content
│   │   └── jobs/
│   │       ├── webhookQueue.js          # Background job processing
│   │       └── schedules.js             # Cron jobs (reminders, reports)
│   ├── migrations/                       # Database schema migrations
│   ├── seeds/                            # Seed data (default programs, etc.)
│   ├── tests/
│   │   ├── unit/
│   │   └── integration/
│   ├── .env.example
│   ├── package.json
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── utils/
│   │   └── App.js
│   ├── public/
│   ├── package.json
│   └── README.md
└── docs/
    ├── API.md
    ├── DATABASE.md
    └── DEPLOYMENT.md
```

## Phase 1.5: Database Schema

### Technology
- **Primary:** PostgreSQL (production)
- **Alternative:** SQLite (local dev)

### Core Tables

#### users
```sql
id                  UUID PRIMARY KEY
email              VARCHAR UNIQUE NOT NULL
password           VARCHAR NOT NULL (bcrypt hashed)
firstName          VARCHAR
lastName           VARCHAR
phone              VARCHAR
role               ENUM (apprentice, supervisor, journeyworker, administrator, super_admin)
organizationId     UUID FOREIGN KEY
status             ENUM (active, pending_approval, inactive)
pinVerified        BOOLEAN DEFAULT false
createdAt          TIMESTAMP
updatedAt          TIMESTAMP
createdBy          UUID FOREIGN KEY (users.id)
```

#### apprentices
```sql
id                  UUID PRIMARY KEY (APP-xxxx format)
userId             UUID FOREIGN KEY NOT NULL
programId          UUID FOREIGN KEY NOT NULL
organizationId     UUID FOREIGN KEY NOT NULL
status             ENUM (active, inactive, completed, archived)
employer           VARCHAR
supervisor         VARCHAR
supervisorEmail    VARCHAR
startDate          DATE
targetCompletionDate DATE
ojtHours           DECIMAL
rtiHours           DECIMAL
qualifiedOjtHours  DECIMAL
supervisionMinutes DECIMAL
wageProgression    TEXT (JSON)
createdAt          TIMESTAMP
updatedAt          TIMESTAMP
createdBy          UUID
```

#### hourLogs
```sql
id                  UUID PRIMARY KEY (LOG-xxxx format)
apprenticeId       UUID FOREIGN KEY NOT NULL
submittedBy        UUID FOREIGN KEY NOT NULL
ojtDomain          VARCHAR (dataCollection, assessmentSupport, skillAcquisition, behaviorReduction, documentation, crisisManagement)
rtiModule          VARCHAR (13 module options)
ojtHours           DECIMAL
rtiHours           DECIMAL
qualifiedHours     DECIMAL
date                DATE
description        TEXT
status             ENUM (pending, approved, rejected)
approvedBy         UUID FOREIGN KEY
rubricScore        INT (1-5)
rubricDomain       VARCHAR (RBT task domain)
rubricNotes        TEXT
evidenceType       VARCHAR (direct, observation, documentation)
supervisionMinutes INT
supervisionType    VARCHAR (direct, indirect)
remediationRequired BOOLEAN
nextSteps          TEXT
submittedAt        TIMESTAMP
approvedAt         TIMESTAMP
createdAt          TIMESTAMP
updatedAt          TIMESTAMP
```

#### programs
```sql
id                  UUID PRIMARY KEY
name               VARCHAR NOT NULL
code               VARCHAR UNIQUE NOT NULL
type               ENUM (registeredApprenticeship, preApprenticeship, workBasedLearning, industryCertification)
targetOjtHours     DECIMAL NOT NULL
targetRtiHours     DECIMAL NOT NULL
partner            VARCHAR
organizationId     UUID FOREIGN KEY
isDefault          BOOLEAN
createdAt          TIMESTAMP
updatedAt          TIMESTAMP
createdBy          UUID
```

#### organizations
```sql
id                  UUID PRIMARY KEY
name               VARCHAR NOT NULL
code               VARCHAR UNIQUE NOT NULL
address            VARCHAR
phone              VARCHAR
websiteUrl         VARCHAR
createdAt          TIMESTAMP
updatedAt          TIMESTAMP
createdBy          UUID
```

#### documents
```sql
id                  UUID PRIMARY KEY
filename           VARCHAR NOT NULL
fileSize           BIGINT
mimeType           VARCHAR
storageUrl         VARCHAR (S3 or local path)
uploadedBy         UUID FOREIGN KEY NOT NULL
organizationId     UUID FOREIGN KEY
apprenticeId       UUID FOREIGN KEY (optional, for apprentice-specific docs)
createdAt          TIMESTAMP
```

## Phase 2: Authentication & Security

### Implementation Details

#### Password Hashing
- Use `bcrypt` with salt rounds: 12
- Never store plaintext passwords
- Implement "forgot password" flow with secure token

#### JWT Tokens
```javascript
{
  id: "user-uuid",
  email: "user@domain.com",
  role: "apprentice|supervisor|journeyworker|administrator|super_admin",
  organizationId: "org-uuid",
  iat: timestamp,
  exp: timestamp + 7 days
}
```

#### Super Admin PIN
- PIN verification screen before accessing super admin features
- PIN (071676) verified against hashed value in environment
- PIN verification token stored in session (not in JWT)
- 6-digit PIN enforced

#### Account Approval Workflow
1. User self-registers → status: "pending_approval"
2. Admin/Super Admin approves → status: "active"
3. Super Admin created accounts → auto-approved, status: "active"

#### Role-Based Access Control (RBAC)

| Role | Features |
|------|----------|
| **Apprentice** | Submit hours, view own progress, competency map |
| **Supervisor** | Review/approve hours, assign rubric scores, view assigned apprentices |
| **Journeyworker** | Same as Supervisor (PA ATO compliance separate role) |
| **Administrator** | Full CRUD on apprentices, view all data, analytics |
| **Super Admin** | Everything + User Mgmt + Program Mgmt + Org Mgmt + Document Mgmt |

### Security Practices
- CORS configuration for frontend domain
- Rate limiting on auth endpoints
- Input validation on all endpoints
- SQL injection prevention (parameterized queries)
- XSS protection via JSON API (no HTML rendering)
- HTTPS only in production
- Secure session management

## Phase 3: API Endpoints

### Auth Endpoints
```
POST   /api/auth/register              # Self-registration
POST   /api/auth/login                 # Login with email/password
POST   /api/auth/logout                # Logout
POST   /api/auth/verify-pin            # Verify super admin PIN
POST   /api/auth/refresh-token         # Refresh JWT
POST   /api/auth/forgot-password       # Request password reset
POST   /api/auth/reset-password        # Reset password with token
```

### User Endpoints (Admin & Super Admin)
```
GET    /api/users                      # List all users (paginated)
GET    /api/users/:id                  # Get user by ID
POST   /api/users                      # Create user (super admin only)
PUT    /api/users/:id                  # Update user (self or admin)
DELETE /api/users/:id                  # Delete user (super admin only)
GET    /api/users/pending              # List pending approval users (admin)
PUT    /api/users/:id/approve          # Approve pending user (admin)
```

### Apprentice Endpoints
```
GET    /api/apprentices                # List all apprentices (role-based filtering)
GET    /api/apprentices/:id            # Get apprentice details
POST   /api/apprentices                # Create apprentice (admin/super admin)
PUT    /api/apprentices/:id            # Update apprentice
DELETE /api/apprentices/:id            # Delete apprentice (soft delete)
GET    /api/apprentices/:id/progress   # Get apprentice progress/dashboard data
GET    /api/apprentices/:id/hours      # Get apprentice hour logs
GET    /api/apprentices/:id/competency # Get competency map/heat map
```

### Hour Log Endpoints
```
POST   /api/hour-logs                  # Submit hours (apprentice)
GET    /api/hour-logs                  # List hour logs (role-based filtering)
GET    /api/hour-logs/:id              # Get hour log details
PUT    /api/hour-logs/:id              # Update hour log (pending only)
PUT    /api/hour-logs/:id/approve      # Approve hours (supervisor/admin)
PUT    /api/hour-logs/:id/reject       # Reject hours (supervisor/admin)
GET    /api/hour-logs/pending          # Get pending approvals for reviewer
```

### Program Endpoints
```
GET    /api/programs                   # List all programs
POST   /api/programs                   # Create program (super admin)
PUT    /api/programs/:id               # Update program (super admin)
DELETE /api/programs/:id               # Delete program (super admin)
GET    /api/programs/:id/details       # Get program with OJT domains, RTI modules
```

### Organization Endpoints (Super Admin)
```
GET    /api/organizations              # List all organizations
POST   /api/organizations              # Create organization
PUT    /api/organizations/:id          # Update organization
DELETE /api/organizations/:id          # Delete organization
```

### Document Endpoints
```
POST   /api/documents                  # Upload document
GET    /api/documents                  # List documents (org-scoped)
DELETE /api/documents/:id              # Delete document
```

### Analytics Endpoints
```
GET    /api/analytics/dashboard        # Dashboard stats (role-based)
GET    /api/analytics/hour-breakdown   # Hour qualification breakdown
GET    /api/analytics/domain-progress  # Progress by OJT domain
GET    /api/analytics/competency-heat-map # RBT task competency data
GET    /api/analytics/supervision-ratio   # Supervision % tracking
```

### Webhook Endpoints
```
POST   /api/webhooks/make              # Make.com webhook receiver
GET    /api/webhooks/status            # Status/health check
```

## Phase 4: Frontend React Refactoring

### Component Structure
```
components/
├── Auth/
│   ├── LoginPage.jsx
│   ├── RegisterPage.jsx
│   ├── PINVerification.jsx
│   ├── ForgotPassword.jsx
│   └── ResetPassword.jsx
├── Dashboard/
│   ├── ApprenticeDashboard.jsx
│   ├── SupervisorDashboard.jsx
│   ├── AdminDashboard.jsx
│   ├── SuperAdminDashboard.jsx
│   └── shared/
│       ├── StatCard.jsx
│       ├── ProgressBar.jsx
│       ├── CompetencyHeatMap.jsx
│       └── SupervisionRatioIndicator.jsx
├── Apprentice/
│   ├── SubmitHoursForm.jsx
│   ├── ProgressOverview.jsx
│   ├── SubmissionHistory.jsx
│   ├── CompetencyMap.jsx
│   └── DomainProgressBars.jsx
├── Hours/
│   ├── ApprovalList.jsx
│   ├── ReviewModal.jsx
│   ├── RubricScoring.jsx
│   ├── CascadingDropdowns.jsx
│   └── RemediationForm.jsx
├── Users/
│   ├── UserManagement.jsx
│   ├── UserForm.jsx
│   ├── ApprovalQueue.jsx
│   └── UserTable.jsx
├── Programs/
│   ├── ProgramManagement.jsx
│   ├── ProgramForm.jsx
│   └── ProgramTable.jsx
├── Layout/
│   ├── Sidebar.jsx (role-based navigation)
│   ├── Header.jsx (user, language toggle)
│   ├── ProtectedRoute.jsx
│   └── RoleGuard.jsx
└── Common/
    ├── Toast.jsx
    ├── Modal.jsx
    ├── LanguageToggle.jsx
    └── LoadingSpinner.jsx
```

### State Management
- React Context API for auth state (user, token, role)
- React Context API for app settings (language, organization)
- SWR or React Query for server state & caching
- Redux only if complexity requires (avoid premature optimization)

### Hooks
```javascript
useAuth()              // Get current user, login, logout
useRole()              // Check current role
useOrganization()      // Get current organization context
useLanguage()          // Get current language & translation function
useApprentice()        // Fetch apprentice data with caching
useHourLogs()          // Fetch hour logs with filtering & pagination
usePagination()        // Generic pagination hook
useFormValidation()    // Client-side validation
```

## Phase 5: Make.com Webhook Integration

### Backend Webhook Handler
- Endpoint: `POST /api/webhooks/make`
- Validates Make.com signature (if available)
- Routes by `action` field (registration, new_apprentice, log_hours)
- Transforms webhook payload to database models
- Queues background jobs for email/SMS

### Webhook Processing Flow
```
1. Receive webhook payload
2. Validate schema
3. Route by action type
4. Create/update database records
5. Queue email/SMS jobs
6. Return 200 OK to Make.com
7. Background jobs process asynchronously
```

### Database-Driven Integrations
- Google Sheets: Backend writes to Sheets API directly instead of via Make.com
- Email: Use Nodemailer or SendGrid for automated emails
- SMS: Use Clearstream API directly with valid credentials

## Phase 6: Database Migrations

### Using Knex.js or Sequelize
- Each schema change as separate migration file
- Migrations with timestamps for version control
- Seeds for default programs, domains, RTI modules
- Rollback capability for each migration

## Phase 7: Testing Strategy

### Unit Tests
- Auth middleware & JWT verification
- ID generation utilities
- Data validation functions
- Role permission checks

### Integration Tests
- Auth flow (register, login, PIN verification)
- CRUD operations for all models
- Role-based access control
- Webhook payload processing
- Database transactions & rollbacks

### Test Coverage Target
- Minimum 70% coverage on critical paths (auth, data access)
- Integration tests for all API endpoints
- Load testing on webhook endpoint

## Phase 8: Deployment

### Environment Configuration
```
NODE_ENV=production
DB_HOST=
DB_USER=
DB_PASSWORD=
DB_NAME=ilead_ams
JWT_SECRET=
SUPER_ADMIN_PIN_HASH=
CLEARSTREAM_API_KEY=
GOOGLE_SHEETS_API_KEY=
SENDGRID_API_KEY=
FRONTEND_URL=
AWS_S3_BUCKET=
AWS_S3_REGION=
```

### Deployment Steps
1. Set up PostgreSQL database
2. Run database migrations
3. Seed default programs & organizations
4. Deploy Node.js backend to Lightsail (or separate server)
5. Update Lightsail nginx config for API routing
6. Deploy React frontend to Lightsail (or Netlify/Vercel)
7. Update Make.com webhook URL if backend address changes
8. Configure CORS for frontend domain

### Rollback Strategy
- Database migration rollback capability
- Git tagging for each production release
- Keep previous version accessible for quick rollback

## Migration from V4 HTML

### Steps
1. **Extract Data Structure:** Parse V4 HTML to understand all data models
2. **Manual Data Export:** If production data exists, export localStorage data as JSON
3. **Data Import:** Create migration script to import V4 JSON data into new PostgreSQL schema
4. **Feature Parity:** Ensure all V4 features work identically in new stack
5. **User Migration:** Create seamless transition plan (parallel running, data sync, cutover date)

### Data Backup Before Migration
- Export all localStorage data from production
- Create PostgreSQL backup before cutover
- Test restore procedures

## Success Criteria

- All V4 features functional in new backend
- Real authentication with bcrypt hashing
- Database persistence (PostgreSQL or SQLite)
- All webhook integrations working
- 70%+ test coverage on critical paths
- Zero data loss during migration
- Sub-second API response times
- Super Admin PIN verification working
- Role-based access control enforced
- Bilingual UI fully functional

## Timeline Estimate

| Phase | Duration | Priority |
|-------|----------|----------|
| 1-2: Setup & Database | 1-2 days | CRITICAL |
| 3-4: API & Frontend | 5-7 days | CRITICAL |
| 5-6: Integrations & Testing | 3-4 days | HIGH |
| 7-8: Deployment & Cutover | 1-2 days | HIGH |
| **Total** | **10-15 days** | — |

## Notes

- Keep V4 as reference during development
- Test each API endpoint thoroughly before frontend integration
- Plan cutover date with hector.torres@i-leadusa.org
- Maintain backup of all production data
- Document all environment variables for handoff
