const Apprentice = require('../models/Apprentice');
const HourLog = require('../models/HourLog');
const db = require('../config/database');

module.exports = {
  dashboard: async (req, res) => {
    try {
      const organizationId = req.user.organizationId;

      const totalApprentices = await Apprentice.countByOrganization(organizationId, 'active');
      const totalUsers = await db('users').where('organization_id', organizationId).count('id as count').first();
      const hours = await Apprentice.getTotalHours(organizationId);

      const pendingLogs = await db('hour_logs')
        .join('apprentices', 'hour_logs.apprentice_id', 'apprentices.id')
        .where('apprentices.organization_id', organizationId)
        .where('hour_logs.status', 'pending')
        .count('hour_logs.id as count')
        .first();

      res.json({
        dashboard: {
          totalUsers: totalUsers.count,
          activeApprentices: totalApprentices,
          totalOjtHours: hours.total_ojt || 0,
          qualifiedOjt: hours.qualified || 0,
          pendingApprovals: pendingLogs.count
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      res.status(500).json({ error: 'Internal Server Error', message: err.message, timestamp: new Date().toISOString() });
    }
  },

  domainProgress: async (req, res) => {
    try {
      const organizationId = req.user.organizationId;

      const domains = await db('hour_logs')
        .join('apprentices', 'hour_logs.apprentice_id', 'apprentices.id')
        .where('apprentices.organization_id', organizationId)
        .where('hour_logs.status', 'approved')
        .groupBy('hour_logs.ojt_domain')
        .select('hour_logs.ojt_domain')
        .sum('hour_logs.qualified_hours as hours')
        .orderBy('hour_logs.ojt_domain');

      res.json({
        domainBreakdown: domains.map(d => ({ domain: d.ojt_domain, qualifiedHours: d.hours || 0 })),
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      res.status(500).json({ error: 'Internal Server Error', message: err.message, timestamp: new Date().toISOString() });
    }
  },

  competencyHeatMap: async (req, res) => {
    try {
      const organizationId = req.user.organizationId;

      const ratings = await db('ratings')
        .join('apprentices', 'ratings.apprentice_id', 'apprentices.id')
        .where('apprentices.organization_id', organizationId)
        .select('ratings.task_domain')
        .avg('ratings.score as avg_score')
        .count('ratings.id as count')
        .groupBy('ratings.task_domain')
        .orderBy('ratings.task_domain');

      res.json({
        competencyMap: ratings.map(r => ({ domain: r.task_domain, avgScore: parseFloat(r.avg_score).toFixed(1), count: r.count })),
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      res.status(500).json({ error: 'Internal Server Error', message: err.message, timestamp: new Date().toISOString() });
    }
  }
};
