const db = require('../config/database');
const { generateUserId } = require('../utils/idGenerator');
const { hashPassword } = require('../utils/password');

class User {
  static async create(userData) {
    const {
      email,
      password,
      firstName,
      lastName,
      phone,
      role,
      organizationId,
      createdBy
    } = userData;

    const passwordHash = await hashPassword(password);

    const [user] = await db('users')
      .insert({
        id: generateUserId(),
        email: email.toLowerCase().trim(),
        password_hash: passwordHash,
        first_name: firstName,
        last_name: lastName,
        phone: phone || null,
        role,
        organization_id: organizationId,
        status: 'pending_approval',
        created_by: createdBy,
        created_at: new Date(),
        updated_at: new Date()
      })
      .returning('*');

    return user;
  }

  static async findById(id) {
    return db('users').where('id', id).first();
  }

  static async findByEmail(email) {
    return db('users').where('email', email.toLowerCase().trim()).first();
  }

  static async findByOrganization(organizationId) {
    return db('users').where('organization_id', organizationId);
  }

  static async list(organizationId, filters = {}) {
    let query = db('users').where('organization_id', organizationId);

    if (filters.role) {
      query = query.where('role', filters.role);
    }

    if (filters.status) {
      query = query.where('status', filters.status);
    }

    if (filters.search) {
      const searchTerm = `%${filters.search}%`;
      query = query.where(qb => {
        qb.whereRaw('email ILIKE ?', [searchTerm])
          .orWhereRaw('first_name ILIKE ?', [searchTerm])
          .orWhereRaw('last_name ILIKE ?', [searchTerm]);
      });
    }

    const limit = Math.min(parseInt(filters.limit) || 50, 100);
    const offset = parseInt(filters.offset) || 0;

    return query
      .orderBy('created_at', 'desc')
      .limit(limit)
      .offset(offset);
  }

  static async update(id, updates) {
    const [user] = await db('users')
      .where('id', id)
      .update({
        ...updates,
        updated_at: new Date()
      })
      .returning('*');

    return user;
  }

  static async updatePassword(id, newPassword) {
    const passwordHash = await hashPassword(newPassword);
    return this.update(id, { password_hash: passwordHash });
  }

  static async approve(id) {
    return this.update(id, { status: 'active' });
  }

  static async deactivate(id) {
    return this.update(id, { status: 'inactive' });
  }

  static async delete(id) {
    return db('users').where('id', id).del();
  }

  static async countByOrganization(organizationId) {
    const [result] = await db('users')
      .where('organization_id', organizationId)
      .count('id as count');

    return result.count;
  }

  static async findPendingApprovals(organizationId) {
    return db('users')
      .where('organization_id', organizationId)
      .where('status', 'pending_approval')
      .orderBy('created_at', 'asc');
  }
}

module.exports = User;
