# Contributing to I-LEAD AMS

Thank you for your interest in contributing to the I-LEAD Apprenticeship Management System! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Report security issues privately
- Focus on the code, not the person

## Development Setup

### Prerequisites

- Node.js 18+
- Docker & Docker Compose
- Git
- PostgreSQL 15 (or use Docker)

### Local Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/AI_Powered_Predictive_CRM_Analytics_System.git
cd AI_Powered_Predictive_CRM_Analytics_System
```

2. **Setup Docker environment**

```bash
cd ilead-ams
./docker-init.sh development
```

Or using Make:

```bash
make init
make dev
make migrate
make seed
```

3. **Verify setup**

```bash
curl http://localhost:3000/health
```

Should return:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "environment": "development"
}
```

## Making Changes

### Create a Feature Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

**Branch naming conventions**:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `hotfix/` - Production hotfixes
- `refactor/` - Code refactoring
- `docs/` - Documentation updates

### Development Workflow

1. **Make code changes**

```bash
# Edit files in your editor
# Changes auto-reload with nodemon in Docker
```

2. **Run tests**

```bash
make test
# or
docker-compose exec api npm test
```

3. **Check code style**

```bash
make lint
# or
docker-compose exec api npm run lint
```

4. **Fix style issues**

```bash
make lint-fix
# or
docker-compose exec api npm run lint:fix
```

## Commit Guidelines

### Commit Message Format

Follow conventional commits:

```
type(scope): subject

body

footer
```

**Types**:
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `style` - Code style (formatting, semicolons, etc.)
- `refactor` - Code refactoring
- `perf` - Performance improvements
- `test` - Adding/updating tests
- `chore` - Build, dependencies, tooling
- `ci` - CI/CD configuration

**Examples**:

```bash
# Good
git commit -m "feat(auth): add two-factor authentication"
git commit -m "fix(hourlog): prevent negative hours submission"
git commit -m "docs(api): update endpoint documentation"
git commit -m "test(apprentice): add progress calculation tests"

# Avoid
git commit -m "fix bug"
git commit -m "updated files"
git commit -m "WIP"
```

### Commit Message Body

Include relevant details:

```
feat(auth): implement JWT token refresh mechanism

- Add refresh token generation and validation
- Update token expiry to 7 days for access, 30 for refresh
- Add middleware for token rotation
- Includes tests for token lifecycle

Closes #123
```

## Pull Request Process

### Before Creating PR

1. **Update branch with main**

```bash
git fetch origin
git rebase origin/main
```

2. **Run full test suite**

```bash
make test
```

3. **Run security checks**

```bash
make lint
npm audit
```

### Creating PR

1. **Push your branch**

```bash
git push origin feature/your-feature-name
```

2. **Create Pull Request on GitHub**

Include:
- **Title**: Clear, descriptive title
- **Description**: What, why, and how
- **Screenshots**: If UI changes
- **Testing**: How to test changes
- **Checklist**:
  - [ ] Tests pass locally
  - [ ] Code style fixed
  - [ ] Documentation updated
  - [ ] No breaking changes (or documented)

### Example PR Description

```markdown
## Description

Add email notifications for hour log approvals to improve communication with apprentices.

## Changes

- Created email service wrapper for Nodemailer
- Added notification controller for approval events
- Implemented email templates (approved, rejected, pending)
- Added configuration for SMTP/SendGrid

## How to Test

1. Submit an hour log
2. Approve it as supervisor
3. Check email inbox for notification

## Related Issues

Closes #456
```

### Review Process

**Automated Checks**:
- Tests must pass
- Coverage must not decrease significantly
- Linting must pass
- Security scanning must not find high-severity issues

**Manual Review**:
- Code quality
- Documentation completeness
- Performance impact
- Security considerations

### Merging

Once approved:
- Squash and merge for feature branches
- Create commit message from PR description
- Delete branch after merge

## Testing

### Running Tests

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

### Writing Tests

Place tests in `__tests__` directory:

```javascript
describe('Feature Name', () => {
  beforeAll(async () => {
    // Setup
  });

  afterAll(async () => {
    // Cleanup
  });

  it('should do something', async () => {
    const result = await someFunction();
    expect(result).toBe(expected);
  });
});
```

### Test Coverage Goals

- **Minimum**: 70% overall
- **Controllers**: 80%+
- **Models**: 90%+
- **Utilities**: 100%

## Code Style

### JavaScript/Node.js

Use ESLint configuration:

```bash
make lint        # Check style
make lint-fix    # Auto-fix
```

**Key rules**:
- 2-space indentation
- Semicolons required
- Single quotes for strings
- No unused variables
- Max line length: 100 characters

### Naming Conventions

```javascript
// Constants
const MAX_FILE_SIZE = 52428800;

// Functions
function getUserById(userId) { }
async function submitHourLog(data) { }

// Classes
class HourLogModel { }

// Private methods (prefix with _)
_validateInput(data) { }
```

### File Organization

```
src/
├── controllers/     # Route handlers
├── models/         # Data models
├── routes/         # Express routes
├── middleware/     # Custom middleware
├── services/       # Business logic
├── utils/          # Utility functions
├── migrations/     # Database migrations
├── seeds/          # Test data
└── __tests__/      # Test files
```

## Documentation

### Code Comments

Document complex logic:

```javascript
// Bad
x = y + 1;  // increment y

// Good
// Calculate next page number for pagination
const nextPageNumber = currentPage + 1;
```

### Function Documentation

```javascript
/**
 * Calculate apprentice progress towards OJT target hours
 * @param {string} apprenticeId - Apprentice identifier
 * @param {number} targetHours - Target OJT hours
 * @returns {Promise<Object>} Progress data with percentage
 * @throws {Error} If apprentice not found
 */
async function getApprenticeProgress(apprenticeId, targetHours) {
  // ...
}
```

### README Updates

Update relevant README sections:
- Installation steps
- Configuration options
- API endpoints
- Database schema changes
- Breaking changes

### API Documentation

Ensure endpoint is documented in API spec:

```markdown
### POST /api/hour-logs/submit

Submit new hour log for approval.

**Request**:
```json
{
  "ojtHours": 8,
  "ojtDomain": "dataCollection",
  "logDate": "2026-03-16",
  "description": "Data collection activities"
}
```

**Response**: 201 Created
```json
{
  "hourLog": { ... },
  "message": "Hours submitted for approval"
}
```

**Errors**: 400 Bad Request, 401 Unauthorized, 404 Not Found
```

## Release Process

### Version Numbering

Use Semantic Versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features
- **PATCH**: Bug fixes

Example: v1.2.3

### Release Checklist

- [ ] All tests passing
- [ ] Code coverage acceptable
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in package.json
- [ ] No security warnings
- [ ] Create GitHub Release with tag
- [ ] Deployment workflow completes

## Troubleshooting

### Docker Issues

```bash
# Clean start
make clean
make init

# Check logs
make logs

# Verify database
docker-compose exec db psql -U postgres -l
```

### Test Failures

```bash
# Clear test database
docker-compose down -v

# Restart and rerun tests
make dev
make test
```

### Git Issues

```bash
# Undo uncommitted changes
git checkout -- .

# Undo last commit (keep changes)
git reset --soft HEAD~1

# View commit history
git log --oneline -10

# Amend last commit
git commit --amend
```

## Resources

- [Project Documentation](./README.md)
- [API Documentation](./API.md)
- [Docker Guide](./ilead-ams/DOCKER.md)
- [Database Schema](./ilead-ams/backend/src/migrations/)
- [CI/CD Workflows](./.github/workflows/README.md)

## Questions?

- Check existing issues and discussions
- Ask in pull request comments
- Review project documentation
- Check GitHub Discussions

## Thank You!

Your contributions help improve I-LEAD AMS for apprentices, supervisors, and administrators worldwide. Thank you for your effort and dedication!
