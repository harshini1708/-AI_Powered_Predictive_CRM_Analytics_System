# Phase 7: Frontend React Integration Plan

## Overview
Migrate the single-file React HTML application (`I-LEAD-AMS-V4.html`) to use the production Node.js backend API with proper frontend architecture, state management, and API integration.

## Current State
- **Frontend**: Single HTML file with React 18 from CDN
- **Data Storage**: localStorage (users, apprentices, hour logs, ratings, documents, programs, organizations)
- **Authentication**: Email/password validation against localStorage
- **Backend**: Production-ready Express.js API with JWT, PostgreSQL, full CRUD operations

## Migration Strategy

### Phase 7A: Frontend Project Setup
1. Create modern React frontend directory structure:
   ```
   ilead-ams/
   ├── frontend/
   │   ├── src/
   │   │   ├── api/                    # API service layer
   │   │   │   ├── client.js           # Axios instance with auth header management
   │   │   │   ├── auth.js             # Auth endpoints
   │   │   │   ├── apprentices.js      # Apprentice endpoints
   │   │   │   ├── hourLogs.js         # Hour log endpoints
   │   │   │   ├── users.js            # User management endpoints
   │   │   │   ├── programs.js         # Program endpoints
   │   │   │   └── analytics.js        # Analytics endpoints
   │   │   ├── context/                # React Context for state management
   │   │   │   ├── AuthContext.js      # Auth state & token management
   │   │   │   └── DataContext.js      # Application data state
   │   │   ├── hooks/                  # Custom hooks
   │   │   │   ├── useAuth.js          # Auth hook
   │   │   │   ├── useApi.js           # API call hook
   │   │   │   └── useLocalStorage.js  # localStorage fallback
   │   │   ├── components/             # Reusable React components
   │   │   │   ├── Auth/               # Auth forms
   │   │   │   ├── Dashboard/          # Dashboard views
   │   │   │   ├── Apprentices/        # Apprentice management
   │   │   │   ├── HourLogs/           # Hour log submission/approval
   │   │   │   ├── Users/              # User management
   │   │   │   └── Common/             # Shared components (navbar, sidebar, etc)
   │   │   ├── pages/                  # Page components
   │   │   ├── App.jsx                 # Main app component
   │   │   ├── index.jsx               # Entry point
   │   │   └── styles/                 # CSS modules
   │   ├── public/
   │   │   └── index.html              # HTML entry point
   │   ├── package.json
   │   ├── vite.config.js              # Vite for fast dev/build
   │   ├── .env.example
   │   └── README.md
   ```

2. Set up build tooling:
   - Vite for fast development and optimized production builds
   - React Router for client-side routing
   - Axios for API calls
   - React Context for state management
   - ESLint and Prettier for code quality

### Phase 7B: API Service Layer
1. Create Axios client with:
   - Base URL configuration from environment
   - JWT token injection in headers
   - Automatic token refresh on 401
   - Error handling and retry logic
   - Request/response interceptors

2. Implement service modules for each API domain:
   - Authentication service (register, login, logout, refresh token)
   - User service (CRUD, list, approve)
   - Apprentice service (list, get, create, update, progress)
   - Hour log service (submit, list, approve, reject)
   - Analytics service (dashboard, domain progress, competency heat map)
   - Program service (list, get)
   - Document service (upload, list, delete)

### Phase 7C: State Management
1. Create AuthContext:
   - Store user data
   - Store access/refresh tokens
   - Handle login/logout
   - Provide useAuth hook

2. Create DataContext:
   - Store apprentices, hour logs, users, programs
   - Provide cache invalidation
   - Handle optimistic updates

3. Custom hooks:
   - useAuth: Get current user and auth functions
   - useApi: Make API calls with loading/error states
   - useLocalStorage: Fallback for offline support

### Phase 7D: Component Migration
Migrate components in order of dependency:
1. Authentication components (Login, Register)
2. Navigation (Navbar, Sidebar)
3. User management (Admin dashboard)
4. Apprentice management (List, create, detail, progress)
5. Hour log management (Submit, list, approve)
6. Analytics views (Dashboard, domain progress, heat map)
7. Document management

### Phase 7E: Features & Enhancements
1. Offline support with service workers
2. Real-time notifications (WebSocket or polling)
3. Data synchronization on reconnect
4. Progressive enhancement
5. Accessibility improvements (WCAG 2.1)

### Phase 7F: Testing & Documentation
1. Unit tests for services
2. Component integration tests
3. End-to-end tests
4. Frontend README with setup instructions
5. API integration documentation

## Backend Requirements Met
✅ Authentication API (register, login, refresh token)
✅ User management API
✅ Apprentice management API
✅ Hour log API
✅ Analytics API
✅ Document upload API
✅ JWT token management
✅ Role-based access control
✅ Error handling and validation
✅ OpenAPI/Swagger documentation

## Development Timeline
- **Phase 7A**: Frontend setup & tooling (1-2 hours)
- **Phase 7B**: API service layer (1-2 hours)
- **Phase 7C**: State management (1 hour)
- **Phase 7D**: Component migration (3-4 hours)
- **Phase 7E**: Features & enhancements (2-3 hours)
- **Phase 7F**: Testing & documentation (1-2 hours)

## Success Criteria
✓ All API endpoints are integrated and working
✓ Frontend fully functional without localStorage dependencies
✓ Authentication flows work end-to-end
✓ All CRUD operations work with backend
✓ Error handling and validation in place
✓ Frontend ready for production deployment
✓ Comprehensive documentation

## Next Steps
1. Review and approve this plan
2. Set up frontend directory structure with Vite
3. Create API service layer
4. Migrate authentication
5. Migrate remaining components
6. Integration testing
7. Production deployment

