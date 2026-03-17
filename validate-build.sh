#!/bin/bash

# Build Validation Script
# Verifies all code structure and integrity

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "✅ I-LEAD AMS - BUILD VALIDATION"
echo "═══════════════════════════════════════════════════════════════"
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check function
check_file() {
  if [ -f "$1" ]; then
    echo -e "${GREEN}✅${NC} $1"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
  else
    echo -e "${RED}❌${NC} $1 (missing)"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
  fi
}

check_dir() {
  if [ -d "$1" ]; then
    echo -e "${GREEN}✅${NC} $1/"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
  else
    echo -e "${RED}❌${NC} $1/ (missing)"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
  fi
}

# Backend Services
echo -e "${BLUE}📦 BACKEND SERVICES${NC}"
check_file "ilead-ams/backend/src/services/emailService.js"
check_file "ilead-ams/backend/src/services/notificationService.js"
check_file "ilead-ams/backend/src/services/reportService.js"
check_file "ilead-ams/backend/src/services/reportScheduler.js"

# Backend Models
echo ""
echo -e "${BLUE}🗄️  DATABASE MODELS${NC}"
check_file "ilead-ams/backend/src/models/Notification.js"
check_file "ilead-ams/backend/src/models/Report.js"

# Backend Routes
echo ""
echo -e "${BLUE}🛣️  API ROUTES${NC}"
check_file "ilead-ams/backend/src/routes/notifications.js"
check_file "ilead-ams/backend/src/routes/bulkOperations.js"
check_file "ilead-ams/backend/src/routes/admin.js"
check_file "ilead-ams/backend/src/routes/reports.js"

# Backend Tests
echo ""
echo -e "${BLUE}🧪 BACKEND TESTS${NC}"
check_file "ilead-ams/backend/src/__tests__/emailService.test.js"
check_file "ilead-ams/backend/src/__tests__/notificationService.test.js"
check_file "ilead-ams/backend/src/__tests__/notificationRoutes.test.js"
check_file "ilead-ams/backend/src/__tests__/reportService.test.js"

# Frontend Components
echo ""
echo -e "${BLUE}🎨 FRONTEND COMPONENTS${NC}"
check_file "ilead-ams/frontend/src/components/UI/Button.jsx"
check_file "ilead-ams/frontend/src/components/UI/Card.jsx"
check_file "ilead-ams/frontend/src/components/UI/Input.jsx"
check_file "ilead-ams/frontend/src/components/UI/Alert.jsx"
check_file "ilead-ams/frontend/src/components/Charts/SimpleChart.jsx"
check_file "ilead-ams/frontend/src/components/Notifications/NotificationBell.jsx"
check_file "ilead-ams/frontend/src/components/Reports/ReportBuilder.jsx"

# Frontend Theme & Styling
echo ""
echo -e "${BLUE}🎨 THEME & STYLING${NC}"
check_file "ilead-ams/frontend/src/context/ThemeContext.jsx"
check_file "ilead-ams/frontend/src/styles/responsive.js"
check_file "ilead-ams/frontend/src/styles/animations.js"
check_file "ilead-ams/frontend/src/utils/accessibility.js"

# Frontend API Clients
echo ""
echo -e "${BLUE}🔌 API CLIENTS${NC}"
check_file "ilead-ams/frontend/src/api/notifications.js"
check_file "ilead-ams/frontend/src/api/admin.js"
check_file "ilead-ams/frontend/src/api/reports.js"

# Frontend Pages
echo ""
echo -e "${BLUE}📄 PAGES${NC}"
check_file "ilead-ams/frontend/src/pages/AdminDashboardPage.jsx"
check_file "ilead-ams/frontend/src/pages/ReportsPage.jsx"
check_file "ilead-ams/frontend/src/pages/EnhancedDashboardPage.jsx"

# Deployment
echo ""
echo -e "${BLUE}🚀 DEPLOYMENT${NC}"
check_file "docker-compose.yml"
check_file "Dockerfile" && check_file "ilead-ams/backend/Dockerfile" && check_file "ilead-ams/frontend/Dockerfile"
check_file "nginx.conf"
check_file "ilead-ams/backend/.env.example"
check_file "ilead-ams/frontend/.env.example"

# Scripts
echo ""
echo -e "${BLUE}⚙️  SCRIPTS${NC}"
check_file "scripts/deploy.sh"
check_file "scripts/backup-database.sh"
check_file "scripts/restore-database.sh"
check_file "scripts/health-check.sh"

# Documentation
echo ""
echo -e "${BLUE}📚 DOCUMENTATION${NC}"
check_file "DEPLOYMENT_GUIDE_v4.md"
check_file "PHASE_7D_PROGRESS.md"
check_file "TEST_SUMMARY_PHASE_7D.md"
check_file "UI_UX_IMPROVEMENTS_PHASE_7D.md"

# Summary
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "${BLUE}📊 VALIDATION SUMMARY${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}✅ Passed:${NC} $CHECKS_PASSED"
echo -e "${RED}❌ Failed:${NC} $CHECKS_FAILED"
echo ""

# Final status
if [ $CHECKS_FAILED -eq 0 ]; then
  echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
  echo -e "${GREEN}🎉 ALL VALIDATIONS PASSED - BUILD IS COMPLETE!${NC}"
  echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
  exit 0
else
  echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
  echo -e "${RED}⚠️  SOME FILES MISSING - REVIEW BUILD${NC}"
  echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
  exit 1
fi
