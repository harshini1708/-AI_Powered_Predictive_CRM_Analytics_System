// Test setup and utilities
const db = require('../config/database');

// Initialize test database before all tests
beforeAll(async () => {
  // Run migrations
  await db.migrate.latest();
});

// Clean up after each test
afterEach(async () => {
  // Truncate tables
  const tables = [
    'hour_logs',
    'ratings',
    'documents',
    'password_reset_tokens',
    'apprentices',
    'programs',
    'users',
    'organizations',
    'audit_logs'
  ];

  for (const table of tables) {
    try {
      await db(table).del();
    } catch (err) {
      // Table might not exist
    }
  }
});

// Close database connection after all tests
afterAll(async () => {
  await db.destroy();
});

// Helper function to create test data
async function createTestOrganization() {
  const [org] = await db('organizations')
    .insert({
      id: require('uuid').v4(),
      name: 'Test Org',
      code: 'TEST-001',
      created_at: new Date(),
      updated_at: new Date()
    })
    .returning('*');
  return org;
}

async function createTestUser(orgId, role = 'apprentice', email = 'test@example.com') {
  const [user] = await db('users')
    .insert({
      id: require('uuid').v4(),
      email,
      password_hash: 'hashed-password',
      first_name: 'Test',
      last_name: 'User',
      phone: '6105551234',
      role,
      organization_id: orgId,
      status: 'active',
      created_at: new Date(),
      updated_at: new Date()
    })
    .returning('*');
  return user;
}

async function createTestProgram(orgId) {
  const [program] = await db('programs')
    .insert({
      id: require('uuid').v4(),
      name: 'Test Program',
      code: 'TST-REG',
      type: 'registeredApprenticeship',
      target_ojt_hours: 2000,
      target_rti_hours: 184,
      partner: 'Test Partner',
      organization_id: orgId,
      is_default: true,
      created_at: new Date(),
      updated_at: new Date()
    })
    .returning('*');
  return program;
}

async function createTestApprentice(userId, programId, orgId) {
  const [apprentice] = await db('apprentices')
    .insert({
      id: 'APP-' + require('uuid').v4().substring(0, 8),
      user_id: userId,
      program_id: programId,
      organization_id: orgId,
      status: 'active',
      ojt_hours: 0,
      rti_hours: 0,
      qualified_ojt_hours: 0,
      supervision_minutes: 0,
      created_at: new Date(),
      updated_at: new Date()
    })
    .returning('*');
  return apprentice;
}

module.exports = {
  createTestOrganization,
  createTestUser,
  createTestProgram,
  createTestApprentice
};
