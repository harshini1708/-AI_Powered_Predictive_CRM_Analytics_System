# I-LEAD AMS (Apprenticeship Management System) - Project Brief for Claude Code

## Owner
**Hector Osvaldo Torres-Sepúlveda (Torres)**
- Director of Operations, I-LEAD Inc.
- Email: hector.torres@i-leadusa.org
- Location: Reading, PA
- Doctoral Candidate, Leadership, Alvernia University

## Organization
**I-LEAD Inc.** - nonprofit adult education and workforce development organization serving primarily Latino and underserved communities in Reading, PA.

## PROJECT OVERVIEW

Single-file React HTML application for managing Registered Apprenticeship, Pre-Apprenticeship, and Work-Based Learning programs. Currently deployed on AWS Lightsail (Ubuntu 24.04, nginx). Built to eventually compete with ApprentiScope ($20/user/month SaaS) and be resold to other organizations.

### Current Production URL
http://98.94.220.9 (static IP on Lightsail)

### Current Version
V4 - single HTML file, 661 lines, ~87,000 characters

## Tech Stack (Current)

* React 18 (CDN, UMD build)
* Babel Standalone v7 (MUST be @babel/standalone, NOT babel-standalone@6 which causes blank page)
* No build tools, no bundler, no backend
* localStorage for all data persistence
* Make.com webhooks for Google Sheets + Gmail + Clearstream SMS integration
* AWS Lightsail Ubuntu 24.04, nginx, $3.50/month
* Netlify as backup deployment (drag-and-drop)

## CRITICAL TECHNICAL NOTES

### Babel Version (CAUSES BLANK PAGE IF WRONG)
```html
<!-- CORRECT - Babel v7 -->
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>

<!-- WRONG - Babel v6, causes blank page with modern JS -->
<script src="https://unpkg.com/babel-standalone@6/babel.min.js"></script>
```

### No Optional Chaining
The code avoids `?.` operators throughout because Babel standalone doesn't always handle them correctly. Use explicit null checks instead.

### No Arrow Functions in Top-Level Declarations
Use `function()` syntax throughout for Babel compatibility. Arrow functions are OK inside JSX event handlers.

### No async/await
Use regular `fetch()` with `.then()` or fire-and-forget pattern for webhooks.

## WEBHOOK ARCHITECTURE

Single webhook URL for all three actions, routed by Make.com Router module:
```
https://hook.us2.make.com/o23utkbnivlz57azfnzvppg11vatqsco
```

### Make.com Scenario Structure
```
Webhook → Router (3 branches)
  ├── Branch 1: filter action="registration" → Google Sheets (ILEAD-Users) → Gmail (welcome) → Gmail (admin notify) → Clearstream SMS (HTTP module)
  ├── Branch 2: filter action="new_apprentice" → Google Sheets (ILEAD-Apprentices)
  └── Branch 3: filter action="log_hours" → Google Sheets (ILEAD-Hour-Logs)
```

### CRITICAL Make.com Notes
* Router filters must compare action field to PLAIN TEXT strings (not purple variable tags)
* Google Sheets: "ILEAD-AMS-Data" workbook with 3 tabs
* Headers must be in Row 1, no trailing spaces, exact camelCase match to webhook payloads
* Gmail connected as hector.torres@i-leadusa.org (Google Workspace)
* Clearstream SMS via HTTP module: endpoint https://api.getclearstream.com/v1/texts, fields `to` and `text_body`

### Webhook Payloads

#### Registration:
```json
{
  "action": "registration",
  "id": "USR-...",
  "role": "apprentice|supervisor|journeyworker|administrator",
  "name": "Full Name",
  "email": "email@domain.com",
  "phone": "6105551234",
  "program": "RBT Registered",
  "programCode": "RBT-REG",
  "timestamp": "ISO8601",
  "orgId": "ILEAD-001"
}
```

#### New Apprentice:
```json
{
  "action": "new_apprentice",
  "id": "APP-...",
  "name": "Full Name",
  "email": "email@domain.com",
  "phone": "",
  "program": "RBT Registered",
  "programCode": "RBT-REG",
  "status": "Active",
  "ojt": 0,
  "rti": 0,
  "qualifiedOjt": 0,
  "supervisor": "Supervisor Name",
  "supervisorEmail": "sup@domain.com",
  "employer": "Company",
  "startDate": "2026-03-12",
  "createdBy": "admin@i-leadusa.org",
  "timestamp": "ISO8601",
  "orgId": "ILEAD-001"
}
```

#### Log Hours (both pending and approved):
```json
{
  "action": "log_hours",
  "id": "LOG-...",
  "apprenticeId": "APP-...",
  "apprenticeName": "Name",
  "apprenticeEmail": "email",
  "ojtHours": 4,
  "rtiHours": 0,
  "qualifiedHours": 4,
  "date": "2026-03-12",
  "status": "pending|approved|rejected",
  "submittedBy": "email",
  "submittedByRole": "apprentice",
  "submittedAt": "ISO8601",
  "rubricDomain": "dataCollection",
  "rubricScore": 4,
  "rubricNotes": "observations...",
  "evidenceType": "direct",
  "nextSteps": "",
  "remediationRequired": false,
  "supervisionMinutes": 30,
  "supervisionType": "direct",
  "description": "activity description",
  "timestamp": "ISO8601",
  "orgId": "ILEAD-001"
}
```

### Google Sheets Headers (Exact)
```
ILEAD-Users: action, id, role, name, email, phone, program, programCode, timestamp, orgId

ILEAD-Apprentices: action, id, name, email, phone, program, programCode, status, ojt, rti, qualifiedOjt, supervisor, supervisorEmail, employer, startDate, createdBy, timestamp, orgId

ILEAD-Hour-Logs: action, id, apprenticeId, apprenticeName, apprenticeEmail, ojtHours, rtiHours, qualifiedHours, date, status, submittedBy, submittedByRole, submittedAt, rubricDomain, rubricScore, rubricNotes, evidenceType, nextSteps, remediationRequired, supervisionMinutes, supervisionType, description, timestamp, orgId
```

## APPLICATION FEATURES (V4)

### Roles (5 levels)
1. **Super Admin** - locked to hector.torres@i-leadusa.org, requires 6-digit PIN (071676), full system control
2. **Administrator** - can add apprentices, view all data, analytics
3. **Supervisor** - reviews/approves hours, assigns rubric scores, sees only assigned apprentices
4. **Journeyworker** - same workflow as supervisor, separate role for PA ATO compliance
5. **Apprentice** - submits hours, views own progress, competency map

### Security Features
* Super Admin PIN verification screen (6-digit: 071676)
* `super_admin` role hidden from ALL dropdowns (registration, add user, edit user)
* Account approval workflow: self-registered accounts get "pending_approval" status, must be approved by admin/super admin
* Users created by super admin are auto-approved
* localStorage tampering protection: non-super emails with super_admin role get downgraded
* `isSuper()` function requires `pinVerified === true`

### Programs (Dynamic, managed by Super Admin)

**Default programs:**
* **RBT Registered** (RBT-REG): 2,000 OJT hours, 184 RTI hours, Partner: I-LEAD Inc.
* **Electrical Registered** (ELEC-REG): 8,000 OJT hours, 1,000 RTI hours, Partner: IBEW 743
* **Sheet Metal Registered** (SM-REG): 8,000 OJT hours, 1,000 RTI hours, Partner: SM Local 19

**Program types:** Registered Apprenticeship, Pre-Apprenticeship, Work-Based Learning, Industry Certification

### RBT Task List (3rd Edition 2026) - All 43 Tasks
* **A:** Professional Conduct & Scope of Practice (6 tasks: A-1 through A-6)
* **B:** Assessment (5 tasks: B-1 through B-5)
* **C:** Skill Acquisition (12 tasks: C-1 through C-12)
* **D:** Behavior Reduction (6 tasks: D-1 through D-6)
* **E:** Documentation & Reporting (6 tasks: E-1 through E-6)
* **F:** Professional Development (8 tasks: F-1 through F-8)

### OJT Domains (2,000 hours total for RBT)
* Data Collection: 250h target
* Assessment Support: 200h target
* Skill Acquisition: 600h target
* Behavior Reduction: 400h target
* Documentation & Compliance: 300h target
* Crisis Management: 250h target

### RTI Modules (184 hours total for RBT)
13 modules: Ethics (16h), Measurement (16h), Assessment (16h), Skill Acquisition (24h), Behavior Reduction (20h), Documentation (12h), Crisis (16h), Communication (12h), Legal (12h), Cultural Competence (12h), Technology (12h), Professional Development (8h), Capstone (8h)

### 5-Point Competency Rubric
1. **Not Yet / Aún No** (red) - Accuracy <50%
2. **Emerging / Emergente** (orange) - Accuracy 50-69%
3. **Developing / En Desarrollo** (yellow) - Accuracy 70-89%, Score 3+ = Qualified OJT
4. **Proficient / Competente** (green) - Accuracy 90%+
5. **Mastery / Dominio** (blue) - Accuracy 95%+, trains peers

### Approval Workflow
1. Apprentice submits hours (OJT domain or RTI module + description)
2. Supervisor/Journeyworker reviews submission
3. Selects RBT Task Domain → Specific Task (cascading dropdowns)
4. Assigns competency score (1-5 rubric)
5. Enters supervision minutes, type (direct/indirect), accuracy data, evidence, context
6. Score < 3: remediation plan required (PA ATO compliance)
7. Score < 4: next steps recommended
8. Score >= 3: hours count as "qualified OJT"
9. Score 1-2: hours count as "practice" (not toward completion)

### BACB Supervision Ratio
* 2.5% direct observation minimum of total OJT hours
* Tracked per apprentice
* Visual indicator (green/red) on dashboards
* Warning alert when below threshold

### Bilingual (EN/ES)
ALL user-facing text has Spanish translations via `t(lang, english, spanish)` helper function or `lang === "es"` ternaries. This includes:
* All sidebar navigation items
* All page titles and subtitles
* All form labels and placeholders
* All button text
* All stat card labels
* All dropdown options (roles show "Apprentice / Aprendiz" etc.)
* All toast messages
* All alert messages

### Views by Role

#### Apprentice Dashboard:
* OJT hours stat with completion %
* Overall progress bar (total/target)
* Domain progress bars (6 domains with individual targets)
* Competency heat map (6 RBT domains, 5-color scale with legend)
* Supervision ratio compliance indicator
* Submit hours form (OJT domain dropdown, RTI module dropdown)
* Submission history table
* Competency assessment history

#### Supervisor/Journeyworker Dashboard:
* Assigned count, pending count, ratings given
* Pending approvals list with review modal
* Full rubric scoring interface with cascading RBT task dropdowns
* My Apprentices table with supervision % per apprentice
* Approved log

#### Admin Dashboard:
* Program count, active apprentices, total OJT, qualified OJT, pending, total ratings
* All Apprentices table with add button
* Analytics: hour qualification stats, domain breakdown progress bars, RBT task competency heat map with legend
* All Hour Logs table

#### Super Admin Dashboard (extends Admin):
* Everything Admin has, plus:
* User Management (add, edit, delete, approve pending accounts)
* Program Management (add, delete programs with custom OJT/RTI targets)
* Organization Management (add, delete organizations for multi-tenant)
* Document Library (upload, view, delete files)
* Purple-themed sidebar
* Total Users and Organizations stats
* Wage tracking fields on apprentice records

## DEPLOYMENT

### AWS Lightsail
* Instance: Ubuntu 24.04 LTS, $3.50/month, 512MB RAM
* Static IP: 98.94.220.9
* Web server: nginx
* File location: /var/www/html/index.html
* SSH: Lightsail browser-based terminal

### Upload Method (Terminal paste DOES NOT WORK for large files)
Use Python HTTP upload server:
```bash
# On Lightsail:
sudo rm /var/www/html/index.html
sudo python3 -c "import http.server,os;exec(\"class H(http.server.SimpleHTTPRequestHandler):\n def do_PUT(self):\n  open('/var/www/html'+self.path,'wb').write(self.rfile.read(int(self.headers['Content-Length'])))\n  self.send_response(201);self.end_headers();os._exit(0)\");http.server.HTTPServer(('0.0.0.0',8888),H).serve_forever()"

# From Windows CMD:
curl -T "C:\Users\Torres\Downloads\I-LEAD-AMS-V4.html" http://98.94.220.9:8888/index.html

# Verify:
wc -l /var/www/html/index.html && wc -c /var/www/html/index.html
sudo systemctl restart nginx
```

Remember to open port 8888 in Lightsail Networking firewall before upload, remove after.

### Clearstream SMS API
* API Key: (regenerate - old one was exposed)
* Endpoint: https://api.getclearstream.com/v1/texts
* Headers: X-Api-Key, Content-Type: application/json
* Body fields: to (array of phone numbers), text_body (message string)

## KNOWN ISSUES / LIMITATIONS

1. No real authentication - password field is cosmetic, only email is checked
2. localStorage only - all data lost if user clears browser cache
3. Single file - 87K chars in one HTML file is hard to maintain
4. No real database - needs backend with PostgreSQL/SQLite for production
5. No file persistence - uploaded documents use URL.createObjectURL (session-only)
6. Clearstream API key was exposed in chat - needs regeneration
7. Lightsail terminal paste - doubles content for files over ~500 lines, must use curl upload

## FUTURE ROADMAP

### Phase 1: Backend (Priority)
* Node.js or Python backend with real authentication (bcrypt password hashing)
* PostgreSQL or SQLite database replacing localStorage
* JWT or session-based auth
* API endpoints for all CRUD operations

### Phase 2: Multi-Tenant
* Organization-scoped data isolation
* Custom branding per organization
* Subscription/billing integration
* Admin panel for managing multiple organizations

### Phase 3: Compliance
* USDOL RAPIDS reporting integration
* WIPS report generation
* PIRL field tracking
* Automated compliance alerts

### Phase 4: Enhanced Features
* Real-time notifications (WebSocket)
* Mobile-responsive improvements
* PDF report generation (progress reports, completion certificates)
* Email notifications for pending approvals
* Wage progression tracking and alerts
* Performance evaluation templates beyond rubric scoring

## WRITING/COMMUNICATION PREFERENCES

* Professional but conversational tone
* APA 7 format for academic content
* No em-dashes (use commas)
* No AI buzzwords: streamline, delve, leverage, seamless
* Varied sentence structure
* Natural human-like flow
* Bilingual EN/ES for all user-facing content
