const Apprentice = require('../models/Apprentice');
const User = require('../models/User');
const Program = require('../models/Program');
const HourLog = require('../models/HourLog');
const db = require('../config/database');

module.exports = {
  // List apprentices (role-based filtering)
  list: async (req, res) => {
    try {
      const { status, limit, offset } = req.query;
      const organizationId = req.user.organizationId;

      let filters = { status: status || undefined, limit, offset };

      // Supervisors see only their apprentices
      if (req.user.role === 'supervisor' || req.user.role === 'journeyworker') {
        filters.supervisorEmail = req.user.email;
      }

      const apprentices = await Apprentice.listByOrganization(organizationId, filters);

      res.json({
        data: apprentices.map(a => ({
          id: a.id,
          name: `${a.first_name || ''} ${a.last_name || ''}`.trim(),
          email: a.email,
          program: a.program,
          supervisor: a.supervisor,
          employer: a.employer,
          ojtHours: a.ojt_hours,
          qualifiedOjtHours: a.qualified_ojt_hours,
          status: a.status
        })),
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('List apprentices error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // Get apprentice details
  getById: async (req, res) => {
    try {
      const { id } = req.params;
      const apprentice = await Apprentice.getWithUserDetails(id);

      if (!apprentice) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'Apprentice not found',
          timestamp: new Date().toISOString()
        });
      }

      // Check authorization
      if (apprentice.organization_id !== req.user.organizationId) {
        return res.status(403).json({
          error: 'Forbidden',
          message: 'Cannot access apprentice from different organization',
          timestamp: new Date().toISOString()
        });
      }

      // Supervisors can only see their apprentices
      if ((req.user.role === 'supervisor' || req.user.role === 'journeyworker') &&
          apprentice.email !== req.user.email &&
          apprentice.supervisor_email !== req.user.email &&
          apprentice.journeyworker_email !== req.user.email) {
        return res.status(403).json({
          error: 'Forbidden',
          message: 'Cannot access apprentice not assigned to you',
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        apprentice: {
          id: apprentice.id,
          userId: apprentice.user_id,
          name: `${apprentice.first_name} ${apprentice.last_name}`,
          email: apprentice.email,
          phone: apprentice.phone,
          program: apprentice.program,
          supervisor: apprentice.supervisor,
          supervisorEmail: apprentice.supervisor_email,
          journeyworker: apprentice.journeyworker,
          journeyworkerEmail: apprentice.journeyworker_email,
          employer: apprentice.employer,
          startDate: apprentice.start_date,
          ojtHours: apprentice.ojt_hours,
          rtiHours: apprentice.rti_hours,
          qualifiedOjtHours: apprentice.qualified_ojt_hours,
          supervisionMinutes: apprentice.supervision_minutes,
          wageStart: apprentice.wage_start,
          wageCurrent: apprentice.wage_current,
          status: apprentice.status,
          createdAt: apprentice.created_at
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Get apprentice error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // Get apprentice progress dashboard
  getProgress: async (req, res) => {
    try {
      const { id } = req.params;
      const apprentice = await Apprentice.getWithUserDetails(id);

      if (!apprentice) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'Apprentice not found',
          timestamp: new Date().toISOString()
        });
      }

      // Get program info
      const program = await db('programs').where('id', apprentice.program_id).first();

      // Get hour totals
      const hourTotals = await HourLog.getTotalHours(id, 'approved');

      // Get domain progress
      const domainHours = await HourLog.getDomainHours(id, 'approved');

      // Calculate completion percentage
      const targetOjt = program ? program.target_ojt_hours : 2000;
      const completionPercent = Math.min(100, (hourTotals.qualified / targetOjt) * 100);

      // Get supervision ratio
      const approvedLogs = await db('hour_logs')
        .where('apprentice_id', id)
        .where('status', 'approved')
        .sum('ojt_hours as total_ojt');

      const directSupervision = await db('hour_logs')
        .where('apprentice_id', id)
        .where('status', 'approved')
        .where('supervision_type', 'direct')
        .sum('supervision_minutes as direct_minutes');

      const totalOjt = approvedLogs[0].total_ojt || 0;
      const directMinutes = directSupervision[0].direct_minutes || 0;
      const directHours = directMinutes / 60;
      const supervisionRatio = totalOjt > 0 ? (directHours / totalOjt) * 100 : 0;

      res.json({
        progress: {
          apprenticeId: id,
          name: `${apprentice.first_name} ${apprentice.last_name}`,
          program: program.name,
          targetOjt: program.target_ojt_hours,
          targetRti: program.target_rti_hours,
          ojtHours: hourTotals.total_ojt || 0,
          rtiHours: hourTotals.total_rti || 0,
          qualifiedOjt: hourTotals.qualified || 0,
          completionPercent: Math.round(completionPercent * 10) / 10,
          supervisionRatio: Math.round(supervisionRatio * 10) / 10,
          supervisionOk: supervisionRatio >= 2.5 || totalOjt < 40,
          domainBreakdown: domainHours.map(d => ({
            domain: d.ojt_domain,
            hours: d.hours || 0
          }))
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Get progress error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // Create apprentice (admin only)
  create: async (req, res) => {
    try {
      const {
        email,
        firstName,
        lastName,
        phone,
        programCode,
        supervisor,
        supervisorEmail,
        journeyworker,
        journeyworkerEmail,
        employer,
        startDate,
        wageStart,
        wageCurrent
      } = req.body;

      // Create or find user
      let user = await User.findByEmail(email);
      if (!user) {
        user = await User.create({
          email,
          password: Math.random().toString(36).slice(2),
          firstName,
          lastName,
          phone,
          role: 'apprentice',
          organizationId: req.user.organizationId,
          createdBy: req.user.id
        });
        await User.approve(user.id);
      }

      // Get program
      const program = await Program.findByCode(programCode, req.user.organizationId);
      if (!program) {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Program not found',
          timestamp: new Date().toISOString()
        });
      }

      const apprentice = await Apprentice.create({
        userId: user.id,
        programId: program.id,
        organizationId: req.user.organizationId,
        employer,
        supervisor,
        supervisorEmail,
        journeyworker,
        journeyworkerEmail,
        startDate,
        wageStart,
        wageCurrent,
        createdBy: req.user.id
      });

      res.status(201).json({
        message: 'Apprentice created',
        apprentice: {
          id: apprentice.id,
          userId: apprentice.user_id,
          name: `${firstName} ${lastName}`,
          email: email,
          program: program.name,
          supervisor: supervisor,
          status: apprentice.status
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Create apprentice error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // Update apprentice
  update: async (req, res) => {
    try {
      const { id } = req.params;
      const {
        supervisor,
        supervisorEmail,
        journeyworker,
        journeyworkerEmail,
        employer,
        wageCurrent,
        status
      } = req.body;

      const apprentice = await Apprentice.findById(id);
      if (!apprentice) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'Apprentice not found',
          timestamp: new Date().toISOString()
        });
      }

      const updates = {};
      if (supervisor !== undefined) updates.supervisor = supervisor;
      if (supervisorEmail !== undefined) updates.supervisor_email = supervisorEmail;
      if (journeyworker !== undefined) updates.journeyworker = journeyworker;
      if (journeyworkerEmail !== undefined) updates.journeyworker_email = journeyworkerEmail;
      if (employer !== undefined) updates.employer = employer;
      if (wageCurrent !== undefined) updates.wage_current = wageCurrent;
      if (status !== undefined) updates.status = status;

      const updatedApprentice = await Apprentice.update(id, updates);

      res.json({
        message: 'Apprentice updated',
        apprentice: {
          id: updatedApprentice.id,
          supervisor: updatedApprentice.supervisor,
          employer: updatedApprentice.employer,
          status: updatedApprentice.status
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Update apprentice error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  }
};
