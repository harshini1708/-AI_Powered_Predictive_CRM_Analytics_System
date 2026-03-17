const db = require('../config/database');
const { generateHourLogId } = require('../utils/idGenerator');

class HourLog {
  static async create(data) {
    const {
      apprenticeId,
      submittedBy,
      ojtDomain,
      rtiModule,
      ojtHours,
      rtiHours,
      logDate,
      description,
      createdBy
    } = data;

    const [log] = await db('hour_logs')
      .insert({
        id: generateHourLogId(),
        apprentice_id: apprenticeId,
        submitted_by: submittedBy,
        ojt_domain: ojtDomain || null,
        rti_module: rtiModule || null,
        ojt_hours: ojtHours || 0,
        rti_hours: rtiHours || 0,
        qualified_hours: 0,
        log_date: logDate,
        description: description || null,
        status: 'pending',
        approved_by: null,
        approved_at: null,
        rubric_score: null,
        rubric_domain: null,
        rubric_notes: null,
        evidence_type: null,
        next_steps: null,
        supervision_minutes: 0,
        supervision_type: null,
        remediation_required: false,
        task_domain: null,
        specific_task: null,
        accuracy_data: null,
        context_vars: null,
        created_at: new Date(),
        updated_at: new Date()
      })
      .returning('*');

    return log;
  }

  static async findById(id) {
    return db('hour_logs').where('id', id).first();
  }

  static async listByApprentice(apprenticeId, filters = {}) {
    let query = db('hour_logs').where('apprentice_id', apprenticeId);

    if (filters.status) {
      query = query.where('status', filters.status);
    }

    const limit = Math.min(parseInt(filters.limit) || 50, 100);
    const offset = parseInt(filters.offset) || 0;

    return query
      .orderBy('log_date', 'desc')
      .limit(limit)
      .offset(offset);
  }

  static async listByOrganization(organizationId, filters = {}) {
    let query = db('hour_logs')
      .join('apprentices', 'hour_logs.apprentice_id', 'apprentices.id')
      .where('apprentices.organization_id', organizationId)
      .select('hour_logs.*');

    if (filters.status) {
      query = query.where('hour_logs.status', filters.status);
    }

    const limit = Math.min(parseInt(filters.limit) || 50, 100);
    const offset = parseInt(filters.offset) || 0;

    return query
      .orderBy('hour_logs.log_date', 'desc')
      .limit(limit)
      .offset(offset);
  }

  static async listPendingBySupervisor(supervisorEmail, organizationId) {
    return db('hour_logs')
      .join('apprentices', 'hour_logs.apprentice_id', 'apprentices.id')
      .where('apprentices.organization_id', organizationId)
      .where('hour_logs.status', 'pending')
      .where(qb => {
        qb.where('apprentices.supervisor_email', supervisorEmail)
          .orWhere('apprentices.journeyworker_email', supervisorEmail);
      })
      .select('hour_logs.*')
      .orderBy('hour_logs.created_at', 'asc');
  }

  static async approve(id, approvalData) {
    const {
      approvedBy,
      rubricScore,
      rubricDomain,
      rubricNotes,
      nextSteps,
      remediationRequired,
      taskDomain,
      specificTask,
      supervisionMinutes,
      supervisionType,
      accuracyData,
      contextVars
    } = approvalData;

    const qualifiedHours = rubricScore >= 3 ? (await this.findById(id)).ojt_hours : 0;

    const [log] = await db('hour_logs')
      .where('id', id)
      .update({
        status: 'approved',
        approved_by: approvedBy,
        approved_at: new Date(),
        rubric_score: rubricScore,
        rubric_domain: rubricDomain,
        rubric_notes: rubricNotes,
        next_steps: nextSteps,
        remediation_required: remediationRequired || false,
        task_domain: taskDomain,
        specific_task: specificTask,
        supervision_minutes: supervisionMinutes || 0,
        supervision_type: supervisionType,
        qualified_hours: qualifiedHours,
        accuracy_data: accuracyData,
        context_vars: contextVars,
        updated_at: new Date()
      })
      .returning('*');

    return log;
  }

  static async reject(id, rejectionNotes, rejectedBy) {
    const [log] = await db('hour_logs')
      .where('id', id)
      .update({
        status: 'rejected',
        approved_by: rejectedBy,
        approved_at: new Date(),
        rubric_notes: rejectionNotes,
        updated_at: new Date()
      })
      .returning('*');

    return log;
  }

  static async countByStatus(apprenticeId, status) {
    const [result] = await db('hour_logs')
      .where('apprentice_id', apprenticeId)
      .where('status', status)
      .count('id as count');

    return result.count;
  }

  static async getTotalHours(apprenticeId, status = 'approved') {
    const [result] = await db('hour_logs')
      .where('apprentice_id', apprenticeId)
      .where('status', status)
      .sum('ojt_hours as total_ojt')
      .sum('rti_hours as total_rti')
      .sum('qualified_hours as qualified');

    return result;
  }

  static async getDomainHours(apprenticeId, status = 'approved') {
    return db('hour_logs')
      .where('apprentice_id', apprenticeId)
      .where('status', status)
      .groupBy('ojt_domain')
      .select('ojt_domain')
      .sum('qualified_hours as hours')
      .orderBy('ojt_domain');
  }

  static async delete(id) {
    return db('hour_logs').where('id', id).del();
  }
}

module.exports = HourLog;
