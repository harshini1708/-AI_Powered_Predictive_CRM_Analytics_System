const db = require('../config/database');
const { generateApprenticeId } = require('../utils/idGenerator');

class Apprentice {
  static async create(data) {
    const {
      userId,
      programId,
      organizationId,
      employer,
      supervisor,
      supervisorEmail,
      journeyworker,
      journeyworkerEmail,
      startDate,
      targetCompletionDate,
      wageStart,
      wageCurrent,
      createdBy
    } = data;

    const [apprentice] = await db('apprentices')
      .insert({
        id: generateApprenticeId(),
        user_id: userId,
        program_id: programId,
        organization_id: organizationId,
        status: 'active',
        employer: employer || null,
        supervisor: supervisor || null,
        supervisor_email: supervisorEmail || null,
        journeyworker: journeyworker || null,
        journeyworker_email: journeyworkerEmail || null,
        start_date: startDate || null,
        target_completion_date: targetCompletionDate || null,
        ojt_hours: 0,
        rti_hours: 0,
        qualified_ojt_hours: 0,
        supervision_minutes: 0,
        wage_start: wageStart || null,
        wage_current: wageCurrent || null,
        wage_progression: null,
        created_by: createdBy,
        created_at: new Date(),
        updated_at: new Date()
      })
      .returning('*');

    return apprentice;
  }

  static async findById(id) {
    return db('apprentices').where('id', id).first();
  }

  static async findByUserId(userId) {
    return db('apprentices').where('user_id', userId).first();
  }

  static async listByOrganization(organizationId, filters = {}) {
    let query = db('apprentices').where('organization_id', organizationId);

    if (filters.status) {
      query = query.where('status', filters.status);
    }

    if (filters.supervisorEmail) {
      query = query.where(qb => {
        qb.where('supervisor_email', filters.supervisorEmail)
          .orWhere('journeyworker_email', filters.supervisorEmail);
      });
    }

    const limit = Math.min(parseInt(filters.limit) || 50, 100);
    const offset = parseInt(filters.offset) || 0;

    return query
      .orderBy('created_at', 'desc')
      .limit(limit)
      .offset(offset);
  }

  static async listBySupervisor(supervisorEmail, organizationId) {
    return db('apprentices')
      .where('organization_id', organizationId)
      .where(qb => {
        qb.where('supervisor_email', supervisorEmail)
          .orWhere('journeyworker_email', supervisorEmail);
      })
      .where('status', 'active');
  }

  static async update(id, updates) {
    const [apprentice] = await db('apprentices')
      .where('id', id)
      .update({
        ...updates,
        updated_at: new Date()
      })
      .returning('*');

    return apprentice;
  }

  static async updateHours(id, ojtHours, rtiHours, qualifiedOjt, supervisionMinutes) {
    return this.update(id, {
      ojt_hours: ojtHours,
      rti_hours: rtiHours,
      qualified_ojt_hours: qualifiedOjt,
      supervision_minutes: supervisionMinutes
    });
  }

  static async deactivate(id) {
    return this.update(id, { status: 'inactive' });
  }

  static async delete(id) {
    return db('apprentices').where('id', id).del();
  }

  static async countByOrganization(organizationId, status = 'active') {
    const [result] = await db('apprentices')
      .where('organization_id', organizationId)
      .where('status', status)
      .count('id as count');

    return result.count;
  }

  static async getTotalHours(organizationId, status = 'active') {
    const [result] = await db('apprentices')
      .where('organization_id', organizationId)
      .where('status', status)
      .sum('ojt_hours as total_ojt')
      .sum('qualified_ojt_hours as qualified_ojt');

    return result;
  }

  static async getWithUserDetails(id) {
    return db('apprentices')
      .join('users', 'apprentices.user_id', 'users.id')
      .select(
        'apprentices.*',
        'users.email',
        'users.first_name',
        'users.last_name',
        'users.phone'
      )
      .where('apprentices.id', id)
      .first();
  }
}

module.exports = Apprentice;
