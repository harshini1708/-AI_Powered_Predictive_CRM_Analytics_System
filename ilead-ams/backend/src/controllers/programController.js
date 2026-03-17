const Program = require('../models/Program');

module.exports = {
  list: async (req, res) => {
    try {
      const { type, limit, offset } = req.query;
      const organizationId = req.user.organizationId;

      const programs = await Program.listByOrganization(organizationId, { type, limit, offset });

      res.json({
        data: programs.map(p => ({
          id: p.id,
          code: p.code,
          name: p.name,
          type: p.type,
          targetOjtHours: p.target_ojt_hours,
          targetRtiHours: p.target_rti_hours,
          partner: p.partner,
          isDefault: p.is_default
        })),
        count: programs.length,
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      res.status(500).json({ error: 'Internal Server Error', message: err.message, timestamp: new Date().toISOString() });
    }
  },

  getById: async (req, res) => {
    try {
      const { id } = req.params;
      const program = await Program.findById(id);

      if (!program) {
        return res.status(404).json({ error: 'Not Found', message: 'Program not found', timestamp: new Date().toISOString() });
      }

      res.json({
        program: {
          id: program.id,
          code: program.code,
          name: program.name,
          type: program.type,
          targetOjtHours: program.target_ojt_hours,
          targetRtiHours: program.target_rti_hours,
          partner: program.partner,
          createdAt: program.created_at
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      res.status(500).json({ error: 'Internal Server Error', message: err.message, timestamp: new Date().toISOString() });
    }
  },

  create: async (req, res) => {
    try {
      const { code, name, type, targetOjtHours, targetRtiHours, partner } = req.body;

      const existing = await Program.findByCode(code, req.user.organizationId);
      if (existing) {
        return res.status(409).json({ error: 'Conflict', message: 'Program code already exists', timestamp: new Date().toISOString() });
      }

      const program = await Program.create({
        code,
        name,
        type,
        targetOjtHours,
        targetRtiHours,
        partner,
        organizationId: req.user.organizationId,
        createdBy: req.user.id
      });

      res.status(201).json({
        message: 'Program created',
        program: { id: program.id, code: program.code, name: program.name },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      res.status(500).json({ error: 'Internal Server Error', message: err.message, timestamp: new Date().toISOString() });
    }
  },

  delete: async (req, res) => {
    try {
      const { id } = req.params;
      const program = await Program.findById(id);

      if (!program) {
        return res.status(404).json({ error: 'Not Found', message: 'Program not found', timestamp: new Date().toISOString() });
      }

      await Program.delete(id);

      res.json({ message: 'Program deleted', timestamp: new Date().toISOString() });
    } catch (err) {
      res.status(500).json({ error: 'Internal Server Error', message: err.message, timestamp: new Date().toISOString() });
    }
  }
};
