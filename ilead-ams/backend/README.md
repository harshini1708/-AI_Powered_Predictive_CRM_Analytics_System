# I-LEAD AMS Backend

Node.js/Express.js backend for the I-LEAD Apprenticeship Management System with PostgreSQL database.

## Quick Start

### Prerequisites
- Node.js 16+
- PostgreSQL 12+
- npm 8+

### Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Create environment file:**
   ```bash
   cp .env.example .env
   ```

3. **Configure database connection in `.env`:**
   ```
   DB_HOST=localhost
   DB_PORT=5432
   DB_USER=postgres
   DB_PASSWORD=your-password
   DB_NAME=ilead_ams
   ```

4. **Create database:**
   ```bash
   createdb ilead_ams
   ```

5. **Run migrations:**
   ```bash
   npm run migrate
   ```

6. **Seed initial data:**
   ```bash
   npm run seed
   ```

7. **Start development server:**
   ```bash
   npm run dev
   ```

Server will run on `http://localhost:3000`

## Project Structure

```
backend/
├── src/
│   ├── server.js              # Express app entry point
│   ├── config/
│   │   └── database.js        # Database connection
│   ├── controllers/           # Route handlers
│   ├── routes/                # API route definitions
│   ├── middleware/            # Express middleware
│   ├── models/                # Data models
│   ├── utils/                 # Utility functions
│   ├── migrations/            # Database migrations
│   └── seeds/                 # Seed data
├── .env.example               # Environment template
├── package.json
└── README.md
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Create new account
- `POST /api/auth/login` - Login with email/password
- `POST /api/auth/logout` - Logout (requires auth)
- `POST /api/auth/verify-pin` - Verify super admin PIN (requires auth)
- `POST /api/auth/refresh-token` - Get new JWT token
- `POST /api/auth/forgot-password` - Request password reset
- `POST /api/auth/reset-password` - Reset password with token
- `GET /api/auth/me` - Get current user info (requires auth)

### Users (Admin/Super Admin only)
- `GET /api/users` - List all users
- `GET /api/users/:id` - Get user by ID
- `POST /api/users` - Create new user
- `PUT /api/users/:id` - Update user
- `DELETE /api/users/:id` - Delete user

### Apprentices
- `GET /api/apprentices` - List apprentices (role-based filtering)
- `GET /api/apprentices/:id` - Get apprentice details
- `GET /api/apprentices/:id/progress` - Get progress dashboard
- `GET /api/apprentices/:id/hours` - Get hour logs
- `GET /api/apprentices/:id/competency` - Get competency map
- `POST /api/apprentices` - Create apprentice (admin only)
- `PUT /api/apprentices/:id` - Update apprentice
- `DELETE /api/apprentices/:id` - Delete apprentice (soft delete)

### Hour Logs
- `GET /api/hour-logs` - List hour logs (role-based filtering)
- `GET /api/hour-logs/:id` - Get hour log details
- `POST /api/hour-logs` - Submit hours (apprentice)
- `PUT /api/hour-logs/:id` - Update pending hour log
- `PUT /api/hour-logs/:id/approve` - Approve hours (supervisor/admin)
- `PUT /api/hour-logs/:id/reject` - Reject hours (supervisor/admin)

### Programs
- `GET /api/programs` - List all programs
- `POST /api/programs` - Create program (super admin)
- `PUT /api/programs/:id` - Update program (super admin)
- `DELETE /api/programs/:id` - Delete program (super admin)

### Organizations (Super Admin only)
- `GET /api/organizations` - List organizations
- `POST /api/organizations` - Create organization
- `PUT /api/organizations/:id` - Update organization
- `DELETE /api/organizations/:id` - Delete organization

### Analytics
- `GET /api/analytics/dashboard` - Dashboard stats (role-based)
- `GET /api/analytics/hour-breakdown` - Hour qualification breakdown
- `GET /api/analytics/domain-progress` - Progress by OJT domain
- `GET /api/analytics/competency-heat-map` - RBT task competency
- `GET /api/analytics/supervision-ratio` - Supervision % tracking

### Webhooks
- `POST /api/webhooks/make` - Make.com webhook receiver
- `GET /api/webhooks/status` - Webhook health check

## Authentication

### JWT Token
User receives JWT token on successful login:
```json
{
  "id": "user-uuid",
  "email": "user@example.com",
  "role": "apprentice|supervisor|journeyworker|administrator|super_admin",
  "organizationId": "org-uuid",
  "pinVerified": false
}
```

Include token in Authorization header:
```
Authorization: Bearer <token>
```

### Super Admin PIN
Super admin users must verify PIN after login:
- POST `/api/auth/verify-pin` with 6-digit PIN
- PIN verification token stored in JWT as `pinVerified: true`

## Database Schema

### Users
- id (UUID)
- email (unique)
- password (bcrypt hashed)
- firstName, lastName, phone
- role (apprentice, supervisor, journeyworker, administrator, super_admin)
- organizationId (FK)
- status (active, pending_approval, inactive)
- created_at, updated_at

### Apprentices
- id (UUID, APP-xxxx format)
- userId (FK)
- programId (FK)
- organizationId (FK)
- status (active, inactive, completed)
- supervisor, supervisorEmail
- employer, startDate
- ojtHours, rtiHours, qualifiedOjtHours
- created_at, updated_at

### Hour Logs
- id (UUID, LOG-xxxx format)
- apprenticeId (FK)
- submittedBy (FK to users)
- ojtDomain, rtiModule
- ojtHours, rtiHours, qualifiedHours
- status (pending, approved, rejected)
- approvedBy (FK to users)
- rubricScore, rubricNotes, nextSteps
- created_at, updated_at

### Programs
- id (UUID)
- code (unique)
- name
- type (registeredApprenticeship, preApprenticeship, workBasedLearning, industryCertification)
- targetOjtHours, targetRtiHours
- partner
- organizationId (FK)
- created_at, updated_at

### Organizations
- id (UUID)
- code (unique)
- name, address, phone
- created_at, updated_at

## Environment Variables

See `.env.example` for full list. Key variables:

```
# Database
DB_HOST=localhost
DB_USER=postgres
DB_PASSWORD=****
DB_NAME=ilead_ams

# Server
NODE_ENV=development
PORT=3000

# JWT
JWT_SECRET=your-secret-key
JWT_EXPIRY=7d

# Super Admin
SUPER_ADMIN_EMAIL=hector.torres@i-leadusa.org
SUPER_ADMIN_PIN_HASH=*****
```

## Running Tests

```bash
npm test                  # Run tests once
npm run test:watch       # Run tests in watch mode
```

## Code Quality

```bash
npm run lint             # Check code style
npm run lint:fix         # Auto-fix style issues
```

## Scripts

```bash
npm start                # Start production server
npm run dev             # Start development server with nodemon
npm run migrate         # Run database migrations
npm run migrate:rollback # Rollback database migration
npm run seed            # Seed initial data
npm test               # Run test suite
npm run test:watch    # Run tests in watch mode
npm run lint          # Lint code
npm run lint:fix      # Auto-fix lint issues
```

## Deployment

### To AWS Lightsail (Ubuntu 24.04)

1. **SSH into Lightsail instance:**
   ```bash
   ssh ubuntu@98.94.220.9
   ```

2. **Install Node.js and PostgreSQL:**
   ```bash
   sudo apt update
   sudo apt install -y nodejs npm postgresql postgresql-contrib
   ```

3. **Clone repository and setup:**
   ```bash
   git clone <repo-url> ilead-ams
   cd ilead-ams/backend
   npm install
   cp .env.example .env
   ```

4. **Create database:**
   ```bash
   sudo -u postgres createdb ilead_ams
   ```

5. **Run migrations:**
   ```bash
   npm run migrate
   npm run seed
   ```

6. **Setup systemd service:**
   Create `/etc/systemd/system/ilead-ams.service`:
   ```ini
   [Unit]
   Description=I-LEAD AMS Backend
   After=network.target postgresql.service

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/ilead-ams/backend
   ExecStart=/usr/bin/node src/server.js
   Restart=always
   RestartSec=10
   StandardOutput=journal
   StandardError=journal
   EnvironmentFile=/home/ubuntu/ilead-ams/backend/.env

   [Install]
   WantedBy=multi-user.target
   ```

7. **Enable and start service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable ilead-ams
   sudo systemctl start ilead-ams
   ```

8. **Setup nginx reverse proxy:**
   ```nginx
   server {
     listen 80;
     server_name 98.94.220.9;

     location /api {
       proxy_pass http://localhost:3000;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection 'upgrade';
       proxy_set_header Host $host;
       proxy_cache_bypass $http_upgrade;
     }
   }
   ```

## Troubleshooting

**Port already in use:**
```bash
# Find process on port 3000
lsof -i :3000
# Kill it
kill -9 <PID>
```

**Database connection failed:**
- Check `.env` credentials
- Verify PostgreSQL is running: `sudo service postgresql status`
- Check database exists: `psql -l`

**JWT errors:**
- Ensure `JWT_SECRET` is set in `.env`
- Check token format in Authorization header

## Architecture Notes

- Uses Knex.js for database migrations and querying
- JWT-based authentication with refresh tokens
- Role-based access control (RBAC) middleware
- Async/await for all async operations
- Express error handling middleware
- Request logging and rate limiting
- CORS enabled for frontend domains

## Contributing

- Follow existing code style
- Run linter before committing: `npm run lint:fix`
- Write tests for new features
- Update this README for API changes

## License

Proprietary - I-LEAD Inc.
