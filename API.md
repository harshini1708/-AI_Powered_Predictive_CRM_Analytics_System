# I-LEAD AMS API Documentation

## Overview

The I-LEAD Apprenticeship Management System provides a comprehensive REST API for managing apprenticeship programs, tracking hours, approving logs, and monitoring apprentice progress.

**Base URL**: `https://api.example.com/api`

**API Version**: 1.0.0

**Authentication**: JWT Bearer Token

## Table of Contents

- [Authentication](#authentication)
- [Error Handling](#error-handling)
- [Endpoints](#endpoints)
  - [Auth](#auth)
  - [Users](#users)
  - [Apprentices](#apprentices)
  - [Hour Logs](#hour-logs)
  - [Programs](#programs)
  - [Analytics](#analytics)
  - [Documents](#documents)
  - [Webhooks](#webhooks)

---

## Authentication

All endpoints (except `/auth/register` and `/auth/login`) require authentication using JWT bearer tokens.

### Getting Started

1. **Register a User**
```bash
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123",
  "firstName": "John",
  "lastName": "Doe",
  "phone": "6105551234",
  "role": "apprentice"
}
```

2. **Login**
```bash
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123"
}

Response:
{
  "accessToken": "eyJhbGci...",
  "refreshToken": "eyJhbGci...",
  "user": { ... }
}
```

3. **Use Access Token**
```bash
GET /api/auth/me
Authorization: Bearer eyJhbGci...
```

### Token Structure

**Access Token** (7 days):
```json
{
  "id": "user-uuid",
  "email": "user@example.com",
  "role": "apprentice",
  "organizationId": "org-uuid",
  "pinVerified": false
}
```

**Refresh Token** (30 days):
```json
{
  "userId": "user-uuid",
  "type": "refresh"
}
```

---

## Error Handling

### Standard Error Response

```json
{
  "error": "Unauthorized",
  "message": "Invalid token",
  "code": "INVALID_TOKEN",
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

### HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | GET request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid input data |
| 401 | Unauthorized | Missing/invalid token |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Duplicate email, etc |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Server Error | Internal error |

### Common Error Codes

| Code | Meaning |
|------|---------|
| `INVALID_TOKEN` | JWT token is invalid |
| `TOKEN_EXPIRED` | JWT token has expired |
| `ACCOUNT_PENDING_APPROVAL` | Account not yet approved |
| `VALIDATION_ERROR` | Input validation failed |
| `UNAUTHORIZED` | User lacks permissions |
| `NOT_FOUND` | Resource doesn't exist |

---

## Endpoints

### Auth

#### Register User

**POST** `/auth/register`

Register a new user account.

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123",
  "firstName": "John",
  "lastName": "Doe",
  "phone": "6105551234",
  "role": "apprentice"
}
```

**Parameters**:
- `email` (string, required): Valid email address
- `password` (string, required): Min 8 characters
- `firstName` (string, required): User's first name
- `lastName` (string, required): User's last name
- `phone` (string, optional): Phone number
- `role` (enum, required): `apprentice`, `supervisor`, `journeyworker`, `administrator`

**Response** (201):
```json
{
  "message": "User account created successfully",
  "user": {
    "id": "user-uuid",
    "email": "user@example.com",
    "firstName": "John",
    "lastName": "Doe",
    "role": "apprentice",
    "status": "pending_approval"
  },
  "accessToken": "eyJhbGci...",
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

**Errors**:
- 400: Validation error (invalid email, weak password)
- 409: Email already exists

---

#### Login

**POST** `/auth/login`

Authenticate user and get access tokens.

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123"
}
```

**Response** (200):
```json
{
  "user": {
    "id": "user-uuid",
    "email": "user@example.com",
    "role": "apprentice",
    "status": "active"
  },
  "accessToken": "eyJhbGci...",
  "refreshToken": "eyJhbGci...",
  "expiresIn": "7d",
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

**Errors**:
- 401: Invalid credentials
- 403: Account not approved

---

#### Get Current User

**GET** `/auth/me`

Get authenticated user information.

**Authentication**: Required (Bearer Token)

**Response** (200):
```json
{
  "user": {
    "id": "user-uuid",
    "email": "user@example.com",
    "firstName": "John",
    "lastName": "Doe",
    "role": "apprentice",
    "organizationId": "org-uuid",
    "status": "active",
    "createdAt": "2026-03-01T10:00:00.000Z"
  },
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

**Errors**:
- 401: Missing or invalid token

---

### Users

#### List Users

**GET** `/users`

List all users in organization (admin only).

**Authentication**: Required (admin role)

**Query Parameters**:
- `role` (string): Filter by role
- `status` (string): Filter by status (active, pending_approval, inactive)
- `search` (string): Search by email or name
- `limit` (number): Results per page (default: 50)
- `offset` (number): Pagination offset

**Response** (200):
```json
{
  "data": [
    {
      "id": "user-uuid",
      "email": "user@example.com",
      "firstName": "John",
      "lastName": "Doe",
      "phone": "6105551234",
      "role": "apprentice",
      "status": "active",
      "createdAt": "2026-03-01T10:00:00.000Z"
    }
  ],
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

#### Get User by ID

**GET** `/users/{id}`

Get specific user details.

**Authentication**: Required

**Parameters**:
- `id` (string, path): User UUID

**Response** (200):
```json
{
  "user": {
    "id": "user-uuid",
    "email": "user@example.com",
    "firstName": "John",
    "lastName": "Doe",
    "phone": "6105551234",
    "role": "apprentice",
    "status": "active",
    "createdAt": "2026-03-01T10:00:00.000Z",
    "updatedAt": "2026-03-15T15:30:00.000Z"
  },
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

**Errors**:
- 404: User not found
- 403: Cannot access user from different organization

---

#### Update User

**PUT** `/users/{id}`

Update user profile.

**Authentication**: Required

**Request Body**:
```json
{
  "firstName": "Jonathan",
  "lastName": "Doe",
  "phone": "6105559999",
  "role": "supervisor"
}
```

**Response** (200):
```json
{
  "message": "User updated successfully",
  "user": {
    "id": "user-uuid",
    "email": "user@example.com",
    "firstName": "Jonathan",
    "role": "supervisor",
    "status": "active"
  },
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

#### Approve User

**POST** `/users/{id}/approve`

Approve pending user (admin only).

**Authentication**: Required (admin role)

**Response** (200):
```json
{
  "message": "User approved successfully",
  "user": {
    "id": "user-uuid",
    "email": "user@example.com",
    "firstName": "John",
    "status": "active"
  },
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

### Apprentices

#### List Apprentices

**GET** `/apprentices`

List apprentices (role-filtered).

**Authentication**: Required

**Query Parameters**:
- `status` (string): active, inactive, completed
- `limit` (number): Results per page
- `offset` (number): Pagination offset

**Response** (200):
```json
{
  "data": [
    {
      "id": "APP-xxx",
      "name": "John Doe",
      "email": "john@example.com",
      "program": "RBT-REG",
      "supervisor": "Jane Smith",
      "employer": "ABC Company",
      "ojtHours": 800,
      "qualifiedOjtHours": 750,
      "status": "active"
    }
  ],
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

#### Get Apprentice

**GET** `/apprentices/{id}`

Get apprentice details.

**Authentication**: Required

**Parameters**:
- `id` (string, path): Apprentice ID

**Response** (200):
```json
{
  "apprentice": {
    "id": "APP-xxx",
    "userId": "user-uuid",
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "6105551234",
    "program": "RBT-REG",
    "supervisor": "Jane Smith",
    "supervisorEmail": "jane@example.com",
    "journeyworker": "Mike Johnson",
    "journeyworkerEmail": "mike@example.com",
    "employer": "ABC Company",
    "startDate": "2026-01-01",
    "ojtHours": 800,
    "rtiHours": 120,
    "qualifiedOjtHours": 750,
    "supervisionMinutes": 1200,
    "wageStart": 15.00,
    "wageCurrent": 16.50,
    "status": "active",
    "createdAt": "2026-01-01T10:00:00.000Z"
  },
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

#### Get Apprentice Progress

**GET** `/apprentices/{id}/progress`

Get apprentice dashboard with progress metrics.

**Authentication**: Required

**Parameters**:
- `id` (string, path): Apprentice ID

**Response** (200):
```json
{
  "progress": {
    "apprenticeId": "APP-xxx",
    "name": "John Doe",
    "program": "RBT-REG",
    "targetOjt": 2000,
    "targetRti": 184,
    "ojtHours": 800,
    "rtiHours": 120,
    "qualifiedOjt": 750,
    "completionPercent": 37.5,
    "supervisionRatio": 3.2,
    "supervisionOk": true,
    "domainBreakdown": [
      {
        "domain": "dataCollection",
        "hours": 200
      },
      {
        "domain": "assessment",
        "hours": 150
      }
    ]
  },
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

#### Create Apprentice

**POST** `/apprentices`

Create new apprentice (admin only).

**Authentication**: Required (admin role)

**Request Body**:
```json
{
  "email": "john@example.com",
  "firstName": "John",
  "lastName": "Doe",
  "phone": "6105551234",
  "programCode": "RBT-REG",
  "supervisor": "Jane Smith",
  "supervisorEmail": "jane@example.com",
  "journeyworker": "Mike Johnson",
  "journeyworkerEmail": "mike@example.com",
  "employer": "ABC Company",
  "startDate": "2026-01-01",
  "wageStart": 15.00,
  "wageCurrent": 16.50
}
```

**Response** (201):
```json
{
  "message": "Apprentice created successfully",
  "apprentice": {
    "id": "APP-xxx",
    "userId": "user-uuid",
    "name": "John Doe",
    "email": "john@example.com",
    "program": "RBT-REG",
    "supervisor": "Jane Smith",
    "status": "active"
  },
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

### Hour Logs

#### Submit Hours

**POST** `/hour-logs/submit`

Submit hours for approval (apprentice only).

**Authentication**: Required

**Request Body**:
```json
{
  "ojtDomain": "dataCollection",
  "rtiModule": "ethics",
  "ojtHours": 8,
  "rtiHours": 2,
  "logDate": "2026-03-16",
  "description": "Data collection and ethics training"
}
```

**Parameters**:
- `ojtDomain` (enum): dataCollection, assessment, skillAcquisition, behaviorReduction, documentation, crisisManagement
- `rtiModule` (enum): ethics, measurement, assessProc, skillStrat, behavTech, etc (13 modules)
- `ojtHours` (number): Hours worked
- `rtiHours` (number): RTI hours
- `logDate` (date): ISO 8601 date

**Response** (201):
```json
{
  "message": "Hours submitted for approval",
  "hourLog": {
    "id": "LOG-xxx",
    "date": "2026-03-16",
    "ojtHours": 8,
    "rtiHours": 2,
    "status": "pending",
    "description": "Data collection and ethics training"
  },
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

**Errors**:
- 400: Invalid domain/module
- 404: Apprentice profile not found

---

#### List Hour Logs

**GET** `/hour-logs`

List hour logs (role-filtered).

**Authentication**: Required

**Query Parameters**:
- `status` (string): pending, approved, rejected
- `limit` (number): Results per page
- `offset` (number): Pagination offset

**Response** (200):
```json
{
  "data": [
    {
      "id": "LOG-xxx",
      "apprenticeId": "APP-xxx",
      "date": "2026-03-16",
      "ojtHours": 8,
      "rtiHours": 2,
      "domain": "dataCollection",
      "module": "ethics",
      "status": "pending",
      "rubricScore": null,
      "createdAt": "2026-03-16T10:00:00.000Z"
    }
  ],
  "count": 1,
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

#### Approve Hours

**POST** `/hour-logs/{id}/approve`

Approve hours with rubric scoring (supervisor/admin).

**Authentication**: Required

**Request Body**:
```json
{
  "rubricScore": 4,
  "rubricDomain": "A",
  "specificTask": "A-1",
  "rubricNotes": "Excellent data collection practices",
  "nextSteps": "Continue current practices",
  "supervisionMinutes": 45,
  "supervisionType": "direct"
}
```

**Parameters**:
- `rubricScore` (number): 1-5 scale (1=developing, 5=exemplary)
- `rubricDomain` (enum): A-F (RBT competency areas)
- `specificTask` (string): Task within domain
- `nextSteps` (string): Required if score < 3
- `supervisionType` (enum): direct, indirect

**Response** (200):
```json
{
  "message": "Hours approved with rubric score",
  "hourLog": {
    "id": "LOG-xxx",
    "status": "approved",
    "rubricScore": 4,
    "qualifiedHours": 8,
    "approvedAt": "2026-03-16T23:30:00.000Z"
  },
  "apprenticeTotals": {
    "ojtHours": 800,
    "qualifiedOjt": 750,
    "supervisionMinutes": 1200
  },
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

**Errors**:
- 400: Invalid rubric score (not 1-5)
- 400: Remediation plan required for score < 3

---

#### Reject Hours

**POST** `/hour-logs/{id}/reject`

Reject hours with feedback.

**Authentication**: Required (supervisor/admin)

**Request Body**:
```json
{
  "rubricNotes": "Hours do not align with stated activities"
}
```

**Response** (200):
```json
{
  "message": "Hours rejected",
  "hourLog": {
    "id": "LOG-xxx",
    "status": "rejected",
    "rubricNotes": "Hours do not align with stated activities",
    "rejectedAt": "2026-03-16T23:30:00.000Z"
  },
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

### Programs

#### List Programs

**GET** `/programs`

List apprenticeship programs.

**Authentication**: Required

**Response** (200):
```json
{
  "data": [
    {
      "id": "PROG-xxx",
      "name": "Registered Apprenticeship (RBT)",
      "code": "RBT-REG",
      "type": "registeredApprenticeship",
      "targetOjtHours": 2000,
      "targetRtiHours": 184,
      "partner": "I-LEAD Inc",
      "isDefault": true
    }
  ],
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

#### Get Program

**GET** `/programs/{id}`

Get program details.

**Authentication**: Required

**Response** (200):
```json
{
  "program": {
    "id": "PROG-xxx",
    "name": "Registered Apprenticeship (RBT)",
    "code": "RBT-REG",
    "type": "registeredApprenticeship",
    "targetOjtHours": 2000,
    "targetRtiHours": 184,
    "partner": "I-LEAD Inc",
    "isDefault": true,
    "createdAt": "2026-01-01T10:00:00.000Z"
  },
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

### Analytics

#### Dashboard Stats

**GET** `/analytics/dashboard`

Get organization-wide dashboard statistics.

**Authentication**: Required

**Response** (200):
```json
{
  "dashboard": {
    "totalUsers": 145,
    "activeApprentices": 38,
    "totalOjtHours": 12800,
    "qualifiedOjt": 11500,
    "pendingApprovals": 7
  },
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

#### Domain Progress

**GET** `/analytics/domain-progress`

Get OJT domain breakdown.

**Authentication**: Required

**Response** (200):
```json
{
  "domainBreakdown": [
    {
      "domain": "dataCollection",
      "qualifiedHours": 3200
    },
    {
      "domain": "assessment",
      "qualifiedHours": 2800
    }
  ],
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

#### Competency Heat Map

**GET** `/analytics/competency-heat-map`

Get rubric score statistics by domain.

**Authentication**: Required

**Response** (200):
```json
{
  "competencyMap": [
    {
      "domain": "A",
      "avgScore": "4.2",
      "count": 85
    },
    {
      "domain": "B",
      "avgScore": "3.8",
      "count": 78
    }
  ],
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

### Documents

#### Upload Document

**POST** `/documents`

Upload a document.

**Authentication**: Required

**Request**: multipart/form-data
- `file` (file): Document file (max 50MB)
- `apprenticeId` (string): Optional apprentice association

**Response** (201):
```json
{
  "document": {
    "id": "DOC-xxx",
    "filename": "apprentice_cert.pdf",
    "fileSize": 245600,
    "mimeType": "application/pdf",
    "storageUrl": "s3://bucket/documents/DOC-xxx/apprentice_cert.pdf",
    "uploadedAt": "2026-03-16T23:30:00.000Z"
  },
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

**Errors**:
- 400: Invalid file type
- 413: File too large

---

#### List Documents

**GET** `/documents`

List documents in organization.

**Authentication**: Required

**Query Parameters**:
- `apprenticeId` (string): Filter by apprentice
- `limit` (number): Results per page

**Response** (200):
```json
{
  "documents": [
    {
      "id": "DOC-xxx",
      "filename": "apprentice_cert.pdf",
      "fileSize": 245600,
      "uploadedAt": "2026-03-16T10:00:00.000Z"
    }
  ],
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

### Webhooks

#### Receive Event

**POST** `/webhooks/make`

Receive events from Make.com automation.

**No Authentication Required**

**Request Body**:
```json
{
  "event": "registration",
  "userId": "user-uuid",
  "email": "user@example.com",
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

**Response** (200):
```json
{
  "success": true,
  "message": "Event logged",
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

## Rate Limiting

All endpoints are rate limited to prevent abuse.

**Default Limits**:
- General endpoints: 100 requests per 15 minutes
- Auth endpoints: 5 failed login attempts per 15 minutes

**Response Headers**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1679097000
```

**When Limit Exceeded**:
```json
{
  "error": "Too Many Requests",
  "message": "Rate limit exceeded, try again later",
  "timestamp": "2026-03-16T23:30:00.000Z"
}
```

---

## Pagination

Endpoints that return lists support pagination.

**Query Parameters**:
- `limit` (number): Items per page (default: 50, max: 100)
- `offset` (number): Number of items to skip (default: 0)

**Response Format**:
```json
{
  "data": [ ... ],
  "pagination": {
    "limit": 50,
    "offset": 0,
    "total": 245
  }
}
```

---

## Sorting

List endpoints support sorting.

**Query Parameter**:
- `sortBy` (string): Field name
- `order` (enum): asc, desc

**Example**:
```
GET /api/apprentices?sortBy=createdAt&order=desc
```

---

## Filtering

Common filters across endpoints.

**Example**:
```
GET /api/hour-logs?status=pending&sortBy=createdAt&limit=20
```

---

## Versions and Compatibility

**Current Version**: 1.0.0

**Versioning**: All endpoints are prefixed with `/api/` (v1 implicit)

**Breaking Changes Policy**:
- Major versions introduce breaking changes
- Deprecation notice provided 30 days before removal
- Old versions supported for 12 months

---

## SDKs and Tools

**Official SDKs**:
- JavaScript/Node.js: `@ilead-ams/sdk-js`
- Python: `ilead-ams-sdk`

**Tools**:
- Postman Collection: [Link to collection](./postman-collection.json)
- OpenAPI Spec: [Link to spec](./openapi.yml)
- API Explorer: https://api.example.com/docs

---

## Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: api-support@i-leadusa.org
- **Slack**: #api-support (if available)

---

**Last Updated**: March 2026
**Maintainer**: I-LEAD Development Team
**License**: See LICENSE file
