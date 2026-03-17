const db = require('../config/database');
const { v4: uuidv4 } = require('uuid');

class Organization {
  static async create(data) {
    const {
      name,
      code,
      address,
      phone,
      websiteUrl,
      createdBy
    } = data;

    const [org] = await db('organizations')
      .insert({
        id: uuidv4(),
        name,
        code: code.toUpperCase(),
        address: address || null,
        phone: phone || null,
        website_url: websiteUrl || null,
        created_at: new Date(),
        updated_at: new Date()
      })
      .returning('*');

    return org;
  }

  static async findById(id) {
    return db('organizations').where('id', id).first();
  }

  static async findByCode(code) {
    return db('organizations').where('code', code.toUpperCase()).first();
  }

  static async list(filters = {}) {
    let query = db('organizations');

    if (filters.search) {
      const searchTerm = `%${filters.search}%`;
      query = query.where(qb => {
        qb.whereRaw('name ILIKE ?', [searchTerm])
          .orWhereRaw('code ILIKE ?', [searchTerm]);
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
    const [org] = await db('organizations')
      .where('id', id)
      .update({
        ...updates,
        updated_at: new Date()
      })
      .returning('*');

    return org;
  }

  static async delete(id) {
    return db('organizations').where('id', id).del();
  }

  static async count() {
    const [result] = await db('organizations').count('id as count');
    return result.count;
  }
}

module.exports = Organization;
