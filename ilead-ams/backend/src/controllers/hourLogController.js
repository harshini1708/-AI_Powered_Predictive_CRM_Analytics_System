const HourLog = require('../models/HourLog');
const Apprentice = require('../models/Apprentice');
const db = require('../config/database');

const OJT_DOMAINS = ['dataCollection', 'assessment', 'skillAcquisition', 'behaviorReduction', 'documentation', 'crisisManagement'];

const RTI_MODULES = ['ethics', 'measurement', 'assessProc', 'skillStrat', 'behavTech', 'docStandards', 'crisis', 'commSkills', 'legal', 'cultural', 'tech', 'profDev', 'capstone'];

const RBT_TASKS = {
  A: { name: 'Professional Conduct & Scope of Practice', tasks: ['A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6'] },
  B: { name: 'Assessment', tasks: ['B-1', 'B-2', 'B-3', 'B-4', 'B-5'] },
  C: { name: 'Skill Acquisition', tasks: ['C-1', 'C-2', 'C-3', 'C-4', 'C-5', 'C-6', 'C-7', 'C-8', 'C-9', 'C-10', 'C-11', 'C-12'] },
  D: { name: 'Behavior Reduction', tasks: ['D-1', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6'] },
  E: { name: 'Documentation & Reporting', tasks: ['E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6'] },
  F: { name: 'Professional Development', tasks: ['F-1', 'F-2', 'F-3', 'F-4', 'F-5', 'F-6', 'F-7', 'F-8'] }
};

module.exports = {
  // Submit hours (apprentice)
  submit: async (req, res) => {
    try {
      const {
        ojtDomain,
        rtiModule,
        ojtHours,
        rtiHours,
        logDate,
        description
      } = req.body;

      // Validate submission
      if (!ojtHours && !rtiHours) {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Must submit OJT or RTI hours',
          timestamp: new Date().toISOString()
        });
      }

      if (ojtHours && !OJT_DOMAINS.includes(ojtDomain)) {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Invalid OJT domain',
          timestamp: new Date().toISOString()
        });
      }

      if (rtiHours && !RTI_MODULES.includes(rtiModule)) {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Invalid RTI module',
          timestamp: new Date().toISOString()
        });
      }

      // Get apprentice profile for current user
      const apprentice = await Apprentice.findByUserId(req.user.id);
      if (!apprentice) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'Apprentice profile not found. Contact administrator.',
          timestamp: new Date().toISOString()
        });
      }

      // Create hour log
      const log = await HourLog.create({
        apprenticeId: apprentice.id,
        submittedBy: req.user.id,
        ojtDomain,
        rtiModule,
        ojtHours: ojtHours || 0,
        rtiHours: rtiHours || 0,
        logDate,
        description,
        createdBy: req.user.id
      });

      res.status(201).json({
        message: 'Hours submitted for approval',
        hourLog: {
          id: log.id,
          date: log.log_date,
          ojtHours: log.ojt_hours,
          rtiHours: log.rti_hours,
          status: log.status,
          description: log.description
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Submit hours error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // List hour logs (role-filtered)
  list: async (req, res) => {
    try {
      const { status, limit, offset } = req.query;
      const organizationId = req.user.organizationId;

      let logs;
      const filters = { status: status || undefined, limit, offset };

      if (req.user.role === 'apprentice') {
        // Apprentices see their own logs
        const apprentice = await Apprentice.findByUserId(req.user.id);
        if (!apprentice) {
          return res.json({ data: [], count: 0, timestamp: new Date().toISOString() });
        }
        logs = await HourLog.listByApprentice(apprentice.id, filters);
      } else {
        // Supervisors see pending for assigned apprentices
        if (req.user.role === 'supervisor' || req.user.role === 'journeyworker') {
          logs = await HourLog.listPendingBySupervisor(req.user.email, organizationId);
        } else {
          // Admins see all
          logs = await HourLog.listByOrganization(organizationId, filters);
        }
      }

      res.json({
        data: logs.map(l => ({
          id: l.id,
          apprenticeId: l.apprentice_id,
          date: l.log_date,
          ojtHours: l.ojt_hours,
          rtiHours: l.rti_hours,
          domain: l.ojt_domain,
          module: l.rti_module,
          status: l.status,
          rubricScore: l.rubric_score,
          createdAt: l.created_at
        })),
        count: logs.length,
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('List hour logs error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // Get hour log details
  getById: async (req, res) => {
    try {
      const { id } = req.params;
      const log = await HourLog.findById(id);

      if (!log) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'Hour log not found',
          timestamp: new Date().toISOString()
        });
      }

      // Authorization check
      const apprentice = await Apprentice.findById(log.apprentice_id);
      if (apprentice.organization_id !== req.user.organizationId) {
        return res.status(403).json({
          error: 'Forbidden',
          message: 'Cannot access log from different organization',
          timestamp: new Date().toISOString()
        });
      }

      // Apprentices can only see their own logs
      if (req.user.role === 'apprentice' && log.submitted_by !== req.user.id) {
        return res.status(403).json({
          error: 'Forbidden',
          message: 'Cannot access another apprentice\'s log',
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        hourLog: {
          id: log.id,
          apprenticeId: log.apprentice_id,
          date: log.log_date,
          ojtHours: log.ojt_hours,
          rtiHours: log.rti_hours,
          qualifiedHours: log.qualified_hours,
          domain: log.ojt_domain,
          module: log.rti_module,
          description: log.description,
          status: log.status,
          rubricScore: log.rubric_score,
          rubricDomain: log.rubric_domain,
          taskDomain: log.task_domain,
          specificTask: log.specific_task,
          supervisionMinutes: log.supervision_minutes,
          supervisionType: log.supervision_type,
          rubricNotes: log.rubric_notes,
          nextSteps: log.next_steps,
          remediationRequired: log.remediation_required,
          createdAt: log.created_at,
          approvedAt: log.approved_at
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Get hour log error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // Approve hours (supervisor/admin)
  approve: async (req, res) => {
    try {
      const { id } = req.params;
      const {
        rubricScore,
        rubricDomain,
        specificTask,
        rubricNotes,
        nextSteps,
        remediationRequired,
        supervisionMinutes,
        supervisionType,
        accuracyData,
        contextVars
      } = req.body;

      const log = await HourLog.findById(id);
      if (!log) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'Hour log not found',
          timestamp: new Date().toISOString()
        });
      }

      // Validate rubric score
      if (!rubricScore || rubricScore < 1 || rubricScore > 5) {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Rubric score must be 1-5',
          timestamp: new Date().toISOString()
        });
      }

      // Validate task domain
      if (!RBT_TASKS[rubricDomain]) {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Invalid RBT task domain (A-F)',
          timestamp: new Date().toISOString()
        });
      }

      // Remediation required if score < 3
      if (rubricScore < 3 && !nextSteps) {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Remediation plan required for scores below 3',
          timestamp: new Date().toISOString()
        });
      }

      // Approve the log
      const approvedLog = await HourLog.approve(id, {
        approvedBy: req.user.id,
        rubricScore,
        rubricDomain,
        rubricNotes,
        nextSteps,
        remediationRequired: rubricScore < 3,
        taskDomain: rubricDomain,
        specificTask,
        supervisionMinutes,
        supervisionType,
        accuracyData,
        contextVars
      });

      // Update apprentice hour totals
      const apprentice = await Apprentice.findById(log.apprentice_id);
      const totalHours = await HourLog.getTotalHours(log.apprentice_id, 'approved');
      const domainHours = await HourLog.getDomainHours(log.apprentice_id, 'approved');

      await Apprentice.updateHours(
        log.apprentice_id,
        totalHours.total_ojt || 0,
        totalHours.total_rti || 0,
        totalHours.qualified || 0,
        apprentice.supervision_minutes + (supervisionMinutes || 0)
      );

      res.json({
        message: 'Hours approved with rubric score',
        hourLog: {
          id: approvedLog.id,
          status: approvedLog.status,
          rubricScore: approvedLog.rubric_score,
          qualifiedHours: approvedLog.qualified_hours,
          approvedAt: approvedLog.approved_at
        },
        apprenticeTotals: {
          ojtHours: totalHours.total_ojt || 0,
          qualifiedOjt: totalHours.qualified || 0,
          supervisionMinutes: apprentice.supervision_minutes + (supervisionMinutes || 0)
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Approve hours error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // Reject hours (supervisor/admin)
  reject: async (req, res) => {
    try {
      const { id } = req.params;
      const { rubricNotes } = req.body;

      const log = await HourLog.findById(id);
      if (!log) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'Hour log not found',
          timestamp: new Date().toISOString()
        });
      }

      const rejectedLog = await HourLog.reject(id, rubricNotes || 'Rejected', req.user.id);

      res.json({
        message: 'Hours rejected',
        hourLog: {
          id: rejectedLog.id,
          status: rejectedLog.status,
          rubricNotes: rejectedLog.rubric_notes,
          rejectedAt: rejectedLog.approved_at
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Reject hours error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  }
};
