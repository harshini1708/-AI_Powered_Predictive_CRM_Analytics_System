const { v4: uuidv4 } = require('uuid');
const bcrypt = require('bcryptjs');

exports.seed = async function(knex) {
  // Truncate tables (delete all existing data)
  await knex('users').del();
  await knex('organizations').del();
  await knex('programs').del();

  // Create organizations
  const orgId = uuidv4();
  await knex('organizations').insert([
    {
      id: orgId,
      name: 'I-LEAD Inc.',
      code: 'ILEAD-001',
      address: 'Reading, PA',
      phone: '610-555-0123',
      website_url: 'https://i-leadusa.org',
      created_at: new Date(),
      updated_at: new Date()
    }
  ]);

  // Hash passwords
  const adminPassword = await bcrypt.hash('password123', 12);
  const superAdminPassword = await bcrypt.hash('password123', 12);

  // Create users
  const superAdminId = uuidv4();
  const adminId = uuidv4();
  const supervisorId = uuidv4();
  const journeyworkerId = uuidv4();
  const apprenticeId = uuidv4();

  await knex('users').insert([
    {
      id: superAdminId,
      email: 'hector.torres@i-leadusa.org',
      password_hash: superAdminPassword,
      first_name: 'Hector',
      last_name: 'Torres',
      phone: '610-555-0100',
      role: 'super_admin',
      organization_id: orgId,
      status: 'active',
      created_by: null,
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: adminId,
      email: 'admin@i-leadusa.org',
      password_hash: adminPassword,
      first_name: 'Admin',
      last_name: 'User',
      phone: '610-555-0101',
      role: 'administrator',
      organization_id: orgId,
      status: 'active',
      created_by: superAdminId,
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: supervisorId,
      email: 'supervisor@i-leadusa.org',
      password_hash: adminPassword,
      first_name: 'Juan',
      last_name: 'Supervisor',
      phone: '610-555-0102',
      role: 'supervisor',
      organization_id: orgId,
      status: 'active',
      created_by: adminId,
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: journeyworkerId,
      email: 'journeyworker@i-leadusa.org',
      password_hash: adminPassword,
      first_name: 'Maria',
      last_name: 'Journeyworker',
      phone: '610-555-0103',
      role: 'journeyworker',
      organization_id: orgId,
      status: 'active',
      created_by: adminId,
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: apprenticeId,
      email: 'apprentice@i-leadusa.org',
      password_hash: adminPassword,
      first_name: 'Luis',
      last_name: 'Apprentice',
      phone: '610-555-0104',
      role: 'apprentice',
      organization_id: orgId,
      status: 'active',
      created_by: adminId,
      created_at: new Date(),
      updated_at: new Date()
    }
  ]);

  // Create default programs
  const programId1 = uuidv4();
  const programId2 = uuidv4();
  const programId3 = uuidv4();

  await knex('programs').insert([
    {
      id: programId1,
      name: 'RBT Registered',
      code: 'RBT-REG',
      type: 'registeredApprenticeship',
      target_ojt_hours: 2000,
      target_rti_hours: 184,
      partner: 'I-LEAD Inc.',
      organization_id: orgId,
      is_default: true,
      created_by: superAdminId,
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: programId2,
      name: 'Electrical Registered',
      code: 'ELEC-REG',
      type: 'registeredApprenticeship',
      target_ojt_hours: 8000,
      target_rti_hours: 1000,
      partner: 'IBEW 743',
      organization_id: orgId,
      is_default: true,
      created_by: superAdminId,
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: programId3,
      name: 'Sheet Metal Registered',
      code: 'SM-REG',
      type: 'registeredApprenticeship',
      target_ojt_hours: 8000,
      target_rti_hours: 1000,
      partner: 'SM Local 19',
      organization_id: orgId,
      is_default: true,
      created_by: superAdminId,
      created_at: new Date(),
      updated_at: new Date()
    }
  ]);

  console.log('✅ Seed data created successfully');
  console.log('\nTest Credentials:');
  console.log('─────────────────────────────────────────');
  console.log('Super Admin:');
  console.log('  Email: hector.torres@i-leadusa.org');
  console.log('  Password: password123');
  console.log('  PIN: 071676');
  console.log('\nAdmin:');
  console.log('  Email: admin@i-leadusa.org');
  console.log('  Password: password123');
  console.log('\nSupervisor:');
  console.log('  Email: supervisor@i-leadusa.org');
  console.log('  Password: password123');
  console.log('\nJourneyworker:');
  console.log('  Email: journeyworker@i-leadusa.org');
  console.log('  Password: password123');
  console.log('\nApprentice:');
  console.log('  Email: apprentice@i-leadusa.org');
  console.log('  Password: password123');
  console.log('─────────────────────────────────────────');
};
