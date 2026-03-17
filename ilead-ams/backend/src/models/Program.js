const db = require('../config/database');
const { v4: uuidv4 } = require('uuid');

class Program {
  static async create(data) {
    const {
      name,
      code,
      type,
      targetOjtHours,
      targetRtiHours,
      partner,
      organizationId,
      isDefault,
      createdBy
    } = data;

    const [program] = await db('programs')
      .insert({
        id: uuidv4(),
        name,
        code: code.toUpperCase(),
        type: type || 'registeredApprenticeship',
        target_ojt_hours: targetOjtHours,
        target_rti_hours: targetRtiHours,
        partner: partner || null,
        organization_id: organizationId,
        is_default: isDefault || false,
        created_by: createdBy,
        created_at: new Date(),
        updated_at: new Date()
      })
      .returning('*');

    return program;
  }

  static async findById(id) {
    return db('programs').where('id', id).first();
  }

  static async findByCode(code, organizationId) {
    return db('programs')
      .where('code', code.toUpperCase())
      .where('organization_id', organizationId)
      .first();
  }

  static async listByOrganization(organizationId, filters = {}) {
    let query = db('programs').where('organization_id', organizationId);

    if (filters.type) {
      query = query.where('type', filters.type);
    }

    const limit = Math.min(parseInt(filters.limit) || 50, 100);
    const offset = parseInt(filters.offset) || 0;

    return query
      .orderBy('created_at', 'desc')
      .limit(limit)
      .offset(offset);
  }

  static async list(filters = {}) {
    let query = db('programs');

    if (filters.organizationId) {
      query = query.where('organization_id', filters.organizationId);
    }

    if (filters.type) {
      query = query.where('type', filters.type);
    }

    const limit = Math.min(parseInt(filters.limit) || 50, 100);
    const offset = parseInt(filters.offset) || 0;

    return query
      .orderBy('created_at', 'desc')
      .limit(limit)
      .offset(offset);
  }

  static async update(id, updates) {
    const [program] = await db('programs')
      .where('id', id)
      .update({
        ...updates,
        updated_at: new Date()
      })
      .returning('*');

    return program;
  }

  static async delete(id) {
    return db('programs').where('id', id).del();
  }

  static async countByOrganization(organizationId) {
    const [result] = await db('programs')
      .where('organization_id', organizationId)
      .count('id as count');

    return result.count;
  }
}

module.exports = Program;
