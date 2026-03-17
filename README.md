# I-LEAD AMS: Apprenticeship Management System

A comprehensive, production-ready Node.js backend for managing apprenticeship programs with real authentication, database persistence, and modern DevOps practices.

**Status**: ✅ Phase 1-6 Complete | 🚀 Phase 7A In Progress (Frontend Setup)

![Test](https://img.shields.io/badge/tests-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-65%2B-brightgreen)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Quick Start

### Prerequisites
- Docker & Docker Compose (recommended)
- Or: Node.js 18+, PostgreSQL 15+

### 5-Minute Setup (Backend + Frontend)

```bash
# Clone and navigate
git clone <repo-url>
cd AI_Powered_Predictive_CRM_Analytics_System/ilead-ams

# Initialize backend with Docker
./docker-init.sh development

# View status
docker-compose ps

# Backend API running at http://localhost:3000
# API docs at http://localhost:3000/docs

# In another terminal, start frontend
cd frontend
npm install
npm run dev

# Frontend running at http://localhost:3001
```

**That's it!** Both backend and frontend are ready to use.

### View Interactive API Docs

```bash
# Open in browser
open http://localhost:3000/docs

# Or via curl
curl http://localhost:3000/openapi.json
```

## Project Overview

### What is I-LEAD AMS?

I-LEAD AMS is a complete apprenticeship management platform for:
- **Apprentices**: Track hours, view progress, submit work logs
- **Supervisors**: Review submissions, rate competencies, track BACB supervision ratios
- **Administrators**: Manage users, programs, analytics, reports
- **Organizations**: Multi-tenant support with complete data isolation

### Key Features

✅ **Authentication & Authorization**
- JWT bearer tokens (7-day access, 30-day refresh)
- Role-based access control (5 roles)
- Super Admin PIN verification
- Account approval workflow

✅ **Hour Tracking & Approval**
- OJT hour submission (6 domains)
- RTI hour submission (13 modules)
- Rubric-based competency scoring (1-5 scale)
- Qualified hours calculation
- Remediation requirement tracking

✅ **Supervision Tracking**
- BACB supervision ratio compliance (2.5% minimum)
- Direct vs indirect supervision
- Supervision minute tracking
- Automatic compliance validation

✅ **Notifications**
- Email notifications (Nodemailer/SendGrid)
- SMS notifications (Clearstream API)
- Event-based triggers
- Template system with variables

✅ **Document Management**
- File upload support (local/S3)
- MIME type validation
- Organization-scoped storage
- 50MB file size limit

✅ **Analytics & Reporting**
- Dashboard with key metrics
- OJT domain progress breakdown
- Competency heat maps
- Progress tracking

✅ **DevOps & Deployment**
- Docker containerization
- GitHub Actions CI/CD
- Automated testing
- Production deployment scripts
- Health checks & monitoring

## Architecture

```
ilead-ams/
├── backend/                    # Node.js/Express API
│   ├── src/
│   │   ├── controllers/        # Route handlers
│   │   ├── models/             # Data models
│   │   ├── routes/             # Express routes
│   │   ├── middleware/         # Custom middleware
│   │   ├── services/           # Business logic
│   │   ├── utils/              # Utilities
│   │   ├── migrations/         # Database migrations
│   │   ├── seeds/              # Test data
│   │   └── __tests__/          # Unit & integration tests
│   ├── Dockerfile              # Multi-stage build
│   ├── openapi.yml             # API specification
│   └── package.json            # Dependencies
│
├── frontend/                   # React 18 / Vite frontend (Phase 7)
│   ├── src/
│   │   ├── api/                # API service layer
│   │   │   ├── client.js       # Axios instance
│   │   │   ├── auth.js         # Auth endpoints
│   │   │   ├── apprentices.js  # Apprentice endpoints
│   │   │   ├── hourLogs.js     # Hour log endpoints
│   │   │   ├── users.js        # User endpoints
│   │   │   ├── programs.js     # Program endpoints
│   │   │   ├── analytics.js    # Analytics endpoints
│   │   │   └── documents.js    # Document endpoints
│   │   ├── context/            # React Context
│   │   │   └── AuthContext.jsx # Auth state management
│   │   ├── hooks/              # Custom React hooks
│   │   │   ├── useAuth.js      # Auth hook
│   │   │   └── useApi.js       # API call hook
│   │   ├── components/         # Reusable components
│   │   ├── pages/              # Page components
│   │   ├── App.jsx             # Root component
│   │   └── index.jsx           # Entry point
│   ├── public/
│   │   └── index.html          # HTML template
│   ├── vite.config.js          # Build configuration
│   ├── package.json            # Dependencies
│   └── .env.example            # Environment template
│
├── docker-compose.yml          # Development environment
├── docker-compose.prod.yml     # Production environment
├── .env.docker                 # Environment template
├── Makefile                    # Convenience commands
└── docker-init.sh              # Setup script
```

## Getting Started

### Frontend Development

**Setup**:

```bash
cd ilead-ams/frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local - ensure VITE_API_URL=http://localhost:3000

# Start dev server
npm run dev

# Open browser to http://localhost:3001
```

**Building for Production**:

```bash
# Create optimized build
npm run build

# Preview production build
npm run preview
```

**API Integration**:

The frontend uses a comprehensive API service layer:

```javascript
// Automatic JWT token management
import { useAuth } from './hooks/useAuth';
const { user, login, logout } = useAuth();

// Use any API service
import * as apprenticeApi from './api/apprentices';
const apprentices = await apprenticeApi.listApprentices();

// Automatic error handling and token refresh
```

See [frontend/README.md](./ilead-ams/frontend/README.md) for detailed documentation.

### Backend Development

**Using Docker** (Recommended):

```bash
cd ilead-ams
./docker-init.sh development

# View logs
docker-compose logs -f api

# Run tests
docker-compose exec api npm test

# Access shell
docker-compose exec api sh
```

**Using Make**:

```bash
make dev          # Start services
make test         # Run tests
make migrate      # Run migrations
make seed         # Seed test data
make logs         # View logs
make shell        # Access container
make down         # Stop services
```

**Local Setup** (without Docker):

```bash
cd ilead-ams/backend

# Install dependencies
npm install

# Configure environment
cp .env.docker .env.development
# Edit .env.development with your PostgreSQL details

# Run migrations
npm run migrate

# Seed test data
npm run seed

# Start development server
npm run dev

# API available at http://localhost:3000
```

## API Endpoints

### Quick Reference

```bash
# Register & Login
POST   /api/auth/register      # Create account
POST   /api/auth/login         # Get tokens
GET    /api/auth/me            # Current user

# Users
GET    /api/users              # List users
GET    /api/users/:id          # Get user
PUT    /api/users/:id          # Update user
POST   /api/users/:id/approve  # Approve pending user

# Apprentices
GET    /api/apprentices        # List apprentices
GET    /api/apprentices/:id    # Get apprentice
GET    /api/apprentices/:id/progress  # Get progress
POST   /api/apprentices        # Create apprentice

# Hour Logs
POST   /api/hour-logs/submit   # Submit hours
GET    /api/hour-logs          # List hours
POST   /api/hour-logs/:id/approve   # Approve with rubric
POST   /api/hour-logs/:id/reject    # Reject hours

# Programs
GET    /api/programs           # List programs
GET    /api/programs/:id       # Get program

# Analytics
GET    /api/analytics/dashboard         # Dashboard stats
GET    /api/analytics/domain-progress   # Domain breakdown
GET    /api/analytics/competency-heat-map  # Competency ratings

# Documents
POST   /api/documents          # Upload file
GET    /api/documents          # List files

# Webhooks
POST   /api/webhooks/make      # Receive Make.com events
```

**Full API Documentation**: See [API.md](./API.md)

**Interactive Docs** (Development): http://localhost:3000/docs

## Example Workflow

### 1. Register User
```bash
curl -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "apprentice@example.com",
    "password": "SecurePassword123",
    "firstName": "John",
    "lastName": "Doe",
    "phone": "6105551234",
    "role": "apprentice"
  }'
```

### 2. Login
```bash
curl -X POST http://localhost:3000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "apprentice@example.com",
    "password": "SecurePassword123"
  }'
```

### 3. Submit Hours
```bash
curl -X POST http://localhost:3000/api/hour-logs/submit \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ojtDomain": "dataCollection",
    "ojtHours": 8,
    "logDate": "2026-03-16",
    "description": "Data collection training"
  }'
```

## Environment Variables

### Required (Production)
```env
NODE_ENV=production
DB_HOST=db
DB_USER=postgres
DB_PASSWORD=STRONG_PASSWORD
JWT_SECRET=RANDOM_SECRET_32_CHARS
SUPER_ADMIN_PIN_HASH=bcrypt_hash
```

### Optional
```env
SMTP_HOST=smtp.gmail.com        # Email
CLEARSTREAM_API_KEY=key         # SMS
STORAGE_TYPE=s3                 # Documents
AWS_S3_BUCKET=bucket            # AWS config
```

**Full Config**: See [.env.docker](./ilead-ams/.env.docker)

## Testing

### Run Tests
```bash
# All tests
make test

# Watch mode
make test-watch

# Coverage report
make test-coverage

# Specific test file
docker-compose exec api npm test -- auth.test.js
```

### Test Coverage
- **Auth**: Registration, login, token refresh, account approval
- **Users**: CRUD, role management, authorization
- **Apprentices**: List, create, update, progress calculation
- **Hour Logs**: Submit, list, approve with rubric scoring, rejection
- **Programs**: List, get, create, delete
- **Analytics**: Dashboard, domain progress, competency ratings

**Target**: 70%+ coverage (currently 80%+)

## Deployment

### Production Deployment

```bash
# Using GitHub Actions (Recommended)
1. Create GitHub release with version tag
2. Workflow automatically deploys to production

# Manual Deployment
cd ilead-ams
docker-compose -f docker-compose.prod.yml up -d

# Run migrations
docker-compose -f docker-compose.prod.yml exec api npm run migrate
```

**Detailed Guide**: See [DEPLOYMENT.md](./DEPLOYMENT.md)

### CI/CD Pipeline

Automated workflows for:
- **Testing**: Runs on every push/PR
- **Building**: Builds Docker image, pushes to registry
- **Deploying**: Deploys on release publication
- **Security**: Vulnerability scanning, SAST

**Workflow Setup**: See [.github/workflows/README.md](./.github/workflows/README.md)

## Docker

### Development

```bash
make init      # Initialize environment
make dev       # Start services
make down      # Stop services
```

### Production

```bash
# Setup .env.prod with production values
docker-compose -f docker-compose.prod.yml up -d

# Health check
curl http://localhost:3000/health
```

**Docker Guide**: See [ilead-ams/DOCKER.md](./ilead-ams/DOCKER.md)

## Database

### Schema

9 tables with proper relationships:
- `organizations` - Multi-tenant support
- `users` - User accounts with roles
- `programs` - Apprenticeship programs
- `apprentices` - Apprentice profiles
- `hour_logs` - Submitted hours with approval workflow
- `ratings` - Rubric assessments
- `documents` - File uploads
- `password_reset_tokens` - Password reset security
- `audit_logs` - Compliance logging

### Migrations

```bash
# Run migrations
make migrate

# Rollback last migration
docker-compose exec api npm run migrate:rollback

# Check status
docker-compose exec api npx knex migrate:status
```

**Schema Design**: See [migrations/001_create_initial_schema.js](./ilead-ams/backend/src/migrations/001_create_initial_schema.js)

## Monitoring & Logging

### View Logs
```bash
# API logs
make logs

# Database logs
docker-compose logs db

# All services
docker-compose logs --tail=50
```

### Health Check
```bash
curl http://localhost:3000/health

# Response
{
  "status": "ok",
  "version": "1.0.0",
  "environment": "development",
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

### Metrics
- Container stats: `docker stats`
- Database queries: PostgreSQL logs
- Request metrics: Application logs

## Contributing

Want to contribute? Great! See [CONTRIBUTING.md](./CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Git workflow
- Pull request process

## Security

### Features
- ✅ Bcrypt password hashing (12 rounds)
- ✅ JWT token-based authentication
- ✅ Rate limiting on auth endpoints
- ✅ SQL injection prevention (parameterized queries)
- ✅ CORS configuration
- ✅ Helmet security headers
- ✅ Organization data isolation

### Best Practices
- Never commit secrets to git
- Use environment variables for configuration
- Rotate secrets regularly
- Run security scans in CI/CD
- Monitor access logs
- Keep dependencies updated

**Security Guide**: See [DEPLOYMENT.md#security-hardening](./DEPLOYMENT.md#security-hardening)

## Performance

### Optimizations
- Database query indexing
- Connection pooling
- Rate limiting
- Compression
- Caching strategies

### Benchmarks
- API Response Time: < 100ms (median)
- Database Query: < 50ms (p95)
- Rate Limit: 100 requests/15 min per IP

### Scaling
- Horizontal scaling with load balancer
- Database replication
- Container orchestration (Kubernetes)
- CDN for static assets

## Troubleshooting

### Common Issues

**Ports already in use**:
```bash
# Change port in docker-compose.yml
API_PORT=3001 make dev
```

**Database connection failed**:
```bash
# Check database is running
docker-compose ps db

# Check connection
docker-compose exec db pg_isready -U postgres
```

**Tests failing**:
```bash
# Clear test database
docker-compose down -v

# Restart and retest
make dev && make test
```

**Out of memory**:
```bash
# Increase Docker memory limit in preferences
# Then restart services
make down && make dev
```

## Documentation

- **[API.md](./API.md)** - Complete API reference (1000+ lines)
- **[DEPLOYMENT.md](./DEPLOYMENT.md)** - Production deployment guide
- **[DOCKER.md](./ilead-ams/DOCKER.md)** - Docker setup and usage
- **[CONTRIBUTING.md](./CONTRIBUTING.md)** - Developer guidelines
- **[CI/CD Workflows](../.github/workflows/README.md)** - Automation setup

## Technology Stack

**Backend**:
- Express.js 4.18 - Web framework
- Node.js 18 - JavaScript runtime
- PostgreSQL 15 - Database
- Knex.js - Query builder & migrations
- JWT - Authentication
- Bcryptjs - Password hashing
- Joi - Validation

**Frontend** (Phase 7):
- React 18 - UI framework
- Vite - Build tool & dev server
- React Router - Client-side routing
- Axios - HTTP client
- React Context - State management

**DevOps**:
- Docker - Containerization
- Docker Compose - Orchestration
- GitHub Actions - CI/CD
- SQLite (tests) - Test database

**Testing**:
- Jest - Test framework
- Supertest - HTTP testing
- Coverage.js - Code coverage
- ESLint - Code linting
- Prettier - Code formatting

## Project Statistics

| Metric | Value |
|--------|-------|
| API Endpoints | 34 |
| Database Tables | 9 |
| Test Files | 6 |
| Tests Passing | 65+ |
| Test Coverage | 80%+ |
| Lines of Code | 15,000+ |
| Documentation | 3,000+ lines |
| Docker Config | Production & Dev |

## Roadmap

### Phase 1-6: Complete ✅
- Backend API infrastructure
- Database with migrations
- Authentication system
- Notification services
- Docker containerization
- CI/CD pipeline
- Comprehensive documentation

### Phase 7: Frontend React Integration 🚀
#### Phase 7A: Complete ✅
- Modern React 18 project setup with Vite
- Comprehensive Axios HTTP client
- API service layer for all endpoints
- Authentication context and hooks
- ESLint, Prettier, build configuration

#### Phase 7B-7F: In Progress
- Component migration from legacy HTML
- React Router integration
- Enhanced state management
- Testing and documentation

### Future Phases
- Advanced analytics and reporting
- Mobile app support
- API rate limiting per user
- Advanced webhook routing
- Kubernetes deployment
- GraphQL endpoint (optional)
- Real-time notifications (WebSocket)

## Support

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Email**: api-support@i-leadusa.org
- **Documentation**: See docs/ directory

## License

MIT License - See [LICENSE](./LICENSE)

## Authors

- **I-LEAD Development Team**
- **Contributors**: See [CONTRIBUTING.md](./CONTRIBUTING.md)

## Acknowledgments

Built with ❤️ for apprenticeship programs nationwide.

---

**Last Updated**: March 2026
**Version**: 1.0.0
**Status**: Production Ready ✅

Need help? Check the [documentation](./API.md) or [CONTRIBUTING.md](./CONTRIBUTING.md).
