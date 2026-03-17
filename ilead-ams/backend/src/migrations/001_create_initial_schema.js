exports.up = async function(knex) {
  // Create Organizations table
  await knex.schema.createTable('organizations', table => {
    table.string('id', 36).primary();
    table.string('name', 255).notNullable();
    table.string('code', 50).notNullable().unique();
    table.text('address');
    table.string('phone', 20);
    table.string('website_url', 255);
    table.timestamps(true, true);
  });

  // Create Users table
  await knex.schema.createTable('users', table => {
    table.string('id', 36).primary();
    table.string('email', 255).notNullable().unique();
    table.string('password_hash', 255).notNullable();
    table.string('first_name', 100);
    table.string('last_name', 100);
    table.string('phone', 20);
    table.enum('role', ['apprentice', 'supervisor', 'journeyworker', 'administrator', 'super_admin']).notNullable();
    table.string('organization_id', 36).notNullable().references('id').inTable('organizations').onDelete('cascade');
    table.enum('status', ['active', 'pending_approval', 'inactive']).notNullable().defaultTo('pending_approval');
    table.string('created_by', 36).references('id').inTable('users').onDelete('set null');
    table.timestamps(true, true);
    table.index('email');
    table.index('organization_id');
    table.index('role');
  });

  // Create Programs table
  await knex.schema.createTable('programs', table => {
    table.string('id', 36).primary();
    table.string('name', 255).notNullable();
    table.string('code', 50).notNullable().unique();
    table.enum('type', ['registeredApprenticeship', 'preApprenticeship', 'workBasedLearning', 'industryCertification']).defaultTo('registeredApprenticeship');
    table.decimal('target_ojt_hours', 10, 2).notNullable();
    table.decimal('target_rti_hours', 10, 2).notNullable();
    table.string('partner', 255);
    table.string('organization_id', 36).notNullable().references('id').inTable('organizations').onDelete('cascade');
    table.boolean('is_default').defaultTo(false);
    table.string('created_by', 36).references('id').inTable('users').onDelete('set null');
    table.timestamps(true, true);
    table.index(['organization_id', 'code']);
  });

  // Create Apprentices table
  await knex.schema.createTable('apprentices', table => {
    table.string('id', 36).primary();
    table.string('user_id', 36).notNullable().references('id').inTable('users').onDelete('cascade');
    table.string('program_id', 36).notNullable().references('id').inTable('programs').onDelete('restrict');
    table.string('organization_id', 36).notNullable().references('id').inTable('organizations').onDelete('cascade');
    table.enum('status', ['active', 'inactive', 'completed', 'archived']).notNullable().defaultTo('active');
    table.string('employer', 255);
    table.string('supervisor', 255);
    table.string('supervisor_email', 255);
    table.string('journeyworker', 255);
    table.string('journeyworker_email', 255);
    table.date('start_date');
    table.date('target_completion_date');
    table.decimal('ojt_hours', 10, 2).defaultTo(0);
    table.decimal('rti_hours', 10, 2).defaultTo(0);
    table.decimal('qualified_ojt_hours', 10, 2).defaultTo(0);
    table.decimal('supervision_minutes', 10, 2).defaultTo(0);
    table.decimal('wage_start', 10, 2);
    table.decimal('wage_current', 10, 2);
    table.text('wage_progression');
    table.string('created_by', 36).references('id').inTable('users').onDelete('set null');
    table.timestamps(true, true);
    table.index(['organization_id', 'status']);
    table.index('supervisor_email');
    table.index('journeyworker_email');
  });

  // Create Hour Logs table
  await knex.schema.createTable('hour_logs', table => {
    table.string('id', 36).primary();
    table.string('apprentice_id', 36).notNullable().references('id').inTable('apprentices').onDelete('cascade');
    table.string('submitted_by', 36).notNullable().references('id').inTable('users').onDelete('restrict');
    table.enum('ojt_domain', ['dataCollection', 'assessment', 'skillAcquisition', 'behaviorReduction', 'documentation', 'crisisManagement']);
    table.string('rti_module', 100);
    table.decimal('ojt_hours', 5, 2).defaultTo(0);
    table.decimal('rti_hours', 5, 2).defaultTo(0);
    table.decimal('qualified_hours', 5, 2).defaultTo(0);
    table.date('log_date').notNullable();
    table.text('description');
    table.enum('status', ['pending', 'approved', 'rejected']).notNullable().defaultTo('pending');
    table.string('approved_by', 36).references('id').inTable('users').onDelete('set null');
    table.timestamp('approved_at');
    table.integer('rubric_score');
    table.string('rubric_domain', 10);
    table.text('rubric_notes');
    table.enum('evidence_type', ['direct', 'observation', 'documentation']);
    table.text('next_steps');
    table.integer('supervision_minutes').defaultTo(0);
    table.enum('supervision_type', ['direct', 'indirect']);
    table.boolean('remediation_required').defaultTo(false);
    table.string('task_domain', 10);
    table.text('specific_task');
    table.text('accuracy_data');
    table.text('context_vars');
    table.timestamps(true, true);
    table.index(['apprentice_id', 'status']);
    table.index(['log_date', 'status']);
    table.index(['ojt_domain', 'status']);
  });

  // Create Ratings table
  await knex.schema.createTable('ratings', table => {
    table.string('id', 36).primary();
    table.string('hour_log_id', 36).notNullable().references('id').inTable('hour_logs').onDelete('cascade');
    table.string('apprentice_id', 36).notNullable().references('id').inTable('apprentices').onDelete('cascade');
    table.string('apprentice_name', 255);
    table.date('assessment_date');
    table.string('task_domain', 10);
    table.text('specific_task');
    table.integer('score');
    table.string('label', 50);
    table.enum('supervision_type', ['direct', 'indirect']);
    table.integer('supervision_minutes').defaultTo(0);
    table.text('accuracy');
    table.text('evidence');
    table.text('next_steps');
    table.text('context');
    table.string('assessed_by', 36).notNullable().references('id').inTable('users').onDelete('restrict');
    table.timestamps(true, true);
    table.index(['apprentice_id', 'task_domain']);
  });

  // Create Documents table
  await knex.schema.createTable('documents', table => {
    table.string('id', 36).primary();
    table.string('filename', 255).notNullable();
    table.bigInteger('file_size');
    table.string('mime_type', 100);
    table.string('storage_url', 500);
    table.string('uploaded_by', 36).notNullable().references('id').inTable('users').onDelete('restrict');
    table.string('organization_id', 36).references('id').inTable('organizations').onDelete('set null');
    table.string('apprentice_id', 36).references('id').inTable('apprentices').onDelete('set null');
    table.timestamps(true, true);
    table.index(['organization_id', 'created_at']);
  });

  // Create Password Reset Tokens table
  await knex.schema.createTable('password_reset_tokens', table => {
    table.string('id', 36).primary();
    table.string('user_id', 36).notNullable().references('id').inTable('users').onDelete('cascade');
    table.string('token', 500).notNullable().unique();
    table.timestamp('expires_at').notNullable();
    table.boolean('used').defaultTo(false);
    table.timestamps(true, true);
    table.index('user_id');
    table.index(['token', 'used']);
  });

  // Create Audit Log table
  await knex.schema.createTable('audit_logs', table => {
    table.string('id', 36).primary();
    table.string('user_id', 36).references('id').inTable('users').onDelete('set null');
    table.string('action', 100).notNullable();
    table.string('resource_type', 100);
    table.string('resource_id', 36);
    table.text('details');
    table.string('ip_address', 45);
    table.timestamps(true, true);
    table.index(['user_id', 'created_at']);
    table.index(['action', 'created_at']);
  });
};

exports.down = async function(knex) {
  await knex.schema.dropTable('audit_logs');
  await knex.schema.dropTable('password_reset_tokens');
  await knex.schema.dropTable('documents');
  await knex.schema.dropTable('ratings');
  await knex.schema.dropTable('hour_logs');
  await knex.schema.dropTable('apprentices');
  await knex.schema.dropTable('programs');
  await knex.schema.dropTable('users');
  await knex.schema.dropTable('organizations');
};
