# I-LEAD AMS Frontend

Modern React frontend for the I-LEAD Apprenticeship Management System, integrated with production Node.js backend API.

**Status**: 🚀 Phase 7A Complete - Project Setup & API Integration Layer Ready

## Features

✅ React 18 with functional components and hooks
✅ Vite for fast development and optimized builds
✅ Axios HTTP client with JWT token management
✅ Auth context for state management
✅ React Router for client-side routing
✅ Comprehensive API service layer
✅ Error handling and retry logic
✅ Offline support ready

## Quick Start

### Prerequisites
- Node.js 16+
- npm 8+
- Running backend API (http://localhost:3000)

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
# Start dev server with hot reload
npm run dev

# Open browser
open http://localhost:3001
```

### Build for Production

```bash
# Build optimized bundle
npm run build

# Preview production build
npm run preview
```

## Environment Configuration

```bash
# Copy example config
cp .env.example .env.local

# Edit .env.local with your API URL
VITE_API_URL=http://localhost:3000
VITE_API_TIMEOUT=30000
```

## Project Structure

```
src/
├── api/                          # API service layer
│   ├── client.js                 # Axios instance with token management
│   ├── auth.js                   # Authentication endpoints
│   ├── apprentices.js            # Apprentice CRUD
│   ├── hourLogs.js               # Hour log operations
│   ├── users.js                  # User management
│   ├── programs.js               # Program endpoints
│   ├── analytics.js              # Analytics endpoints
│   └── documents.js              # File upload/download
│
├── context/                      # React Context for state
│   └── AuthContext.jsx           # Authentication state
│
├── hooks/                        # Custom React hooks
│   ├── useAuth.js                # Auth context hook
│   └── useApi.js                 # API call hook
│
├── components/                   # Reusable React components
│   ├── Auth/                     # Login, Register, PIN verify
│   ├── Dashboard/                # Dashboard views
│   ├── Apprentices/              # Apprentice management
│   ├── HourLogs/                 # Hour log submission/approval
│   ├── Users/                    # User administration
│   └── Common/                   # Navbar, Sidebar, etc
│
├── pages/                        # Page-level components
├── App.jsx                       # Main app component
├── App.css                       # Global styles
└── index.jsx                     # React DOM entry point
```

## API Integration

### Axios Client Configuration

The Axios client is pre-configured with:
- **Base URL**: Automatically points to backend API
- **JWT Token Injection**: Automatically adds auth headers
- **Token Refresh**: Handles 401 responses with automatic refresh
- **Error Handling**: Consistent error responses
- **Request/Response Interceptors**: Logging and transformation

### Using the API Services

```javascript
import { useAuth } from './hooks/useAuth';
import * as apprenticeApi from './api/apprentices';

function MyComponent() {
  const { user } = useAuth();

  const handleFetchApprentices = async () => {
    try {
      const response = await apprenticeApi.listApprentices({ limit: 50 });
      console.log('Apprentices:', response.data);
    } catch (error) {
      console.error('Error:', error.response?.data?.message);
    }
  };

  return <button onClick={handleFetchApprentices}>Load Apprentices</button>;
}
```

### Authentication Context

```javascript
import { useAuth } from './hooks/useAuth';

function LoginPage() {
  const { login, error, loading } = useAuth();

  const handleLogin = async (email, password) => {
    try {
      await login(email, password);
      // Redirect to dashboard
    } catch (err) {
      console.error('Login failed:', err);
    }
  };

  return (
    <form onSubmit={(e) => {
      e.preventDefault();
      handleLogin('user@example.com', 'password');
    }}>
      {error && <div className="alert alert-danger">{error}</div>}
      {/* Form inputs */}
    </form>
  );
}
```

### API Calls with Loading States

```javascript
import { useApi } from './hooks/useApi';
import * as analyticsApi from './api/analytics';

function Dashboard() {
  const { data, loading, error, execute } = useApi(analyticsApi.getDashboardStats);

  useEffect(() => {
    execute();
  }, [execute]);

  if (loading) return <div>Loading...</div>;
  if (error) return <div className="alert alert-danger">{error}</div>;

  return (
    <div>
      <h2>Dashboard</h2>
      {data && <p>Total Users: {data.dashboard.totalUsers}</p>}
    </div>
  );
}
```

## API Endpoints

All endpoints are available through service modules:

### Authentication
```javascript
import * as authApi from './api/auth';

authApi.register(userData)
authApi.login(email, password)
authApi.getCurrentUser()
authApi.verifyPin(pin)
authApi.refreshToken(refreshToken)
authApi.logout()
authApi.forgotPassword(email)
authApi.resetPassword(token, password)
```

### Apprentices
```javascript
import * as apprenticeApi from './api/apprentices';

apprenticeApi.listApprentices(filters)
apprenticeApi.getApprentice(id)
apprenticeApi.getApprenticeProgress(id)
apprenticeApi.createApprentice(data)
apprenticeApi.updateApprentice(id, updates)
```

### Hour Logs
```javascript
import * as hourLogApi from './api/hourLogs';

hourLogApi.submitHours(data)
hourLogApi.listHourLogs(filters)
hourLogApi.getHourLog(id)
hourLogApi.approveHourLog(id, approvalData)
hourLogApi.rejectHourLog(id, rejectionData)
```

### Analytics
```javascript
import * as analyticsApi from './api/analytics';

analyticsApi.getDashboardStats()
analyticsApi.getDomainProgress()
analyticsApi.getCompetencyHeatMap()
analyticsApi.getApprenticeAnalytics(id)
analyticsApi.getProgressReport(id)
analyticsApi.getSupervisionReport()
```

### Users
```javascript
import * as userApi from './api/users';

userApi.listUsers(filters)
userApi.getUser(id)
userApi.updateUser(id, updates)
userApi.deleteUser(id)
userApi.approveUser(id)
userApi.createUser(data)
```

### Programs
```javascript
import * as programApi from './api/programs';

programApi.listPrograms()
programApi.getProgram(id)
programApi.createProgram(data)
programApi.updateProgram(id, updates)
programApi.deleteProgram(id)
```

### Documents
```javascript
import * as documentApi from './api/documents';

documentApi.uploadDocument(file, metadata)
documentApi.listDocuments(filters)
documentApi.getDocument(id)
documentApi.downloadDocument(id)
documentApi.deleteDocument(id)
```

## Development Workflow

### Running Tests (When Implemented)

```bash
# Unit tests
npm run test

# Watch mode
npm run test:watch

# Coverage
npm run test:coverage
```

### Code Quality

```bash
# Lint code
npm run lint

# Fix linting issues
npm run lint:fix

# Format code
npm run format
```

## Token Management

The Axios client automatically handles JWT tokens:

1. **Login**: Stores access and refresh tokens
2. **Requests**: Injects access token in Authorization header
3. **401 Response**: Automatically refreshes token and retries
4. **Logout**: Clears both tokens

Tokens are stored in localStorage:
- `accessToken`: Short-lived (7 days)
- `refreshToken`: Long-lived (30 days)

## Error Handling

All API calls include error handling:

```javascript
try {
  const result = await someApi.call();
} catch (error) {
  // error.response.data contains:
  // - error: Error code
  // - message: User-friendly message
  // - code: Error code for programmatic handling
}
```

## Production Deployment

### Build Configuration

Vite automatically optimizes the build with:
- Code splitting (vendor bundle)
- Minification
- Asset optimization
- Source maps (development only)

### Deployment Steps

```bash
# 1. Build for production
npm run build

# 2. Deploy dist/ folder to web server
# Options:
# - Docker container
# - Static hosting (S3, Netlify, Vercel)
# - Web server (nginx, Apache)

# 3. Configure backend API URL
# Set VITE_API_URL in production environment

# 4. Configure CORS on backend
# Allow frontend domain in CORS_ALLOWED_ORIGINS
```

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Performance Optimization

- Code splitting by route
- Lazy component loading
- Optimized bundle size (~100KB gzipped)
- Efficient state management with Context
- Request caching and deduplication

## Offline Support (Future)

The frontend is structured to support:
- Service workers for offline mode
- LocalStorage fallback
- Data synchronization on reconnect
- Offline-first architecture ready

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

## Troubleshooting

### CORS Errors

If you get CORS errors:
1. Ensure backend is running (`npm run dev` in backend)
2. Check VITE_API_URL matches backend URL
3. Verify backend CORS_ALLOWED_ORIGINS includes frontend origin

### Token Issues

If tokens aren't being set:
1. Check browser localStorage: `localStorage.getItem('accessToken')`
2. Verify login response includes tokens
3. Check network tab for failed refresh requests

### Build Issues

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear vite cache
rm -rf dist .vite

# Rebuild
npm run build
```

## License

MIT - See LICENSE file

## Support

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Email**: api-support@i-leadusa.org

---

**Version**: 1.0.0
**Last Updated**: March 2026
**Status**: Development (Phase 7A Complete - API Integration Ready)
