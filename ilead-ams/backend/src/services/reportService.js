import Report from '../models/Report.js';
import HourLog from '../models/HourLog.js';
import User from '../models/User.js';
import Apprentice from '../models/Apprentice.js';

/**
 * Report Service - Handles report generation, scheduling, and delivery
 */
class ReportService {
  /**
   * Create new report
   */
  async createReport(userId, reportData) {
    try {
      const report = new Report({
        ...reportData,
        createdBy: userId,
        owner: userId
      });

      await report.save();
      console.log(`✅ Report created: ${report._id}`);
      return report;
    } catch (error) {
      console.error('Error creating report:', error);
      throw error;
    }
  }

  /**
   * Get user's reports
   */
  async getUserReports(userId, filters = {}) {
    try {
      const { status = 'active', type = null, limit = 20, skip = 0 } = filters;

      const query = {
        $or: [
          { owner: userId },
          { createdBy: userId },
          { 'sharedWith.userId': userId }
        ],
        status
      };

      if (type) query.type = type;

      const reports = await Report
        .find(query)
        .sort({ createdAt: -1 })
        .limit(limit)
        .skip(skip)
        .populate('createdBy', 'firstName lastName email')
        .populate('owner', 'firstName lastName email')
        .lean();

      const total = await Report.countDocuments(query);

      return { reports, total };
    } catch (error) {
      console.error('Error fetching reports:', error);
      throw error;
    }
  }

  /**
   * Get single report
   */
  async getReport(reportId, userId) {
    try {
      const report = await Report.findById(reportId)
        .populate('createdBy', 'firstName lastName email')
        .populate('owner', 'firstName lastName email');

      if (!report) {
        throw new Error('Report not found');
      }

      // Check authorization
      const canView = 
        report.owner._id.toString() === userId ||
        report.createdBy._id.toString() === userId ||
        report.isPublic ||
        report.sharedWith.some(s => s.userId.toString() === userId);

      if (!canView) {
        throw new Error('Unauthorized');
      }

      return report;
    } catch (error) {
      console.error('Error fetching report:', error);
      throw error;
    }
  }

  /**
   * Generate report data
   */
  async generateReportData(report) {
    try {
      const { filters, type, columns, groupBy, sortBy } = report;

      // Build query
      const query = {};

      if (filters.dateRange?.startDate || filters.dateRange?.endDate) {
        query.date = {};
        if (filters.dateRange.startDate) {
          query.date.$gte = new Date(filters.dateRange.startDate);
        }
        if (filters.dateRange.endDate) {
          query.date.$lte = new Date(filters.dateRange.endDate);
        }
      }

      if (filters.apprenticeIds?.length > 0) {
        query.apprenticeId = { $in: filters.apprenticeIds };
      }

      if (filters.status?.length > 0) {
        query.status = { $in: filters.status };
      }

      // Fetch data based on report type
      let data = [];

      switch (type) {
        case 'hours_summary':
          data = await this.generateHoursSummary(query);
          break;
        case 'apprentice_progress':
          data = await this.generateApprenticeProgress(query);
          break;
        case 'supervisor_workload':
          data = await this.generateSupervisorWorkload(query);
          break;
        case 'program_statistics':
          data = await this.generateProgramStatistics(query);
          break;
        case 'completion_forecast':
          data = await this.generateCompletionForecast(query);
          break;
        case 'custom':
          data = await this.generateCustomReport(query, columns);
          break;
      }

      // Apply grouping
      if (groupBy?.field && groupBy.field !== 'none') {
        data = this.groupData(data, groupBy.field);
      }

      // Apply sorting
      if (sortBy?.field) {
        data = this.sortData(data, sortBy.field, sortBy.order);
      }

      return {
        data,
        rowCount: data.length,
        generatedAt: new Date()
      };
    } catch (error) {
      console.error('Error generating report:', error);
      throw error;
    }
  }

  /**
   * Generate hours summary report
   */
  async generateHoursSummary(query) {
    const logs = await HourLog.find(query)
      .populate('apprenticeId', 'firstName lastName')
      .lean();

    return logs.map(log => ({
      date: log.date,
      apprentice: `${log.apprenticeId.firstName} ${log.apprenticeId.lastName}`,
      ojtHours: log.ojtHours || 0,
      rtiHours: log.rtiHours || 0,
      totalHours: (log.ojtHours || 0) + (log.rtiHours || 0),
      status: log.status,
      approvedHours: log.qualifiedHours || 0
    }));
  }

  /**
   * Generate apprentice progress report
   */
  async generateApprenticeProgress(query) {
    const apprentices = await Apprentice.find(query).lean();

    return Promise.all(
      apprentices.map(async (apprentice) => {
        const logs = await HourLog.find({
          apprenticeId: apprentice._id,
          status: 'approved'
        });

        const totalHours = logs.reduce((sum, log) => {
          return sum + (log.qualifiedHours || 0);
        }, 0);

        const progressPercent = (totalHours / (apprentice.requiredHours || 2000)) * 100;

        return {
          apprenticeName: `${apprentice.firstName} ${apprentice.lastName}`,
          program: apprentice.program,
          requiredHours: apprentice.requiredHours || 2000,
          completedHours: totalHours,
          progressPercent: Math.round(progressPercent),
          hoursRemaining: Math.max(0, (apprentice.requiredHours || 2000) - totalHours)
        };
      })
    );
  }

  /**
   * Generate supervisor workload report
   */
  async generateSupervisorWorkload(query) {
    const supervisors = await User.find({ role: 'supervisor' }).lean();

    return Promise.all(
      supervisors.map(async (supervisor) => {
        const pending = await HourLog.countDocuments({
          supervisorId: supervisor._id,
          status: 'pending',
          ...query
        });

        const approved = await HourLog.countDocuments({
          supervisorId: supervisor._id,
          status: 'approved',
          ...query
        });

        return {
          supervisorName: `${supervisor.firstName} ${supervisor.lastName}`,
          pendingApprovals: pending,
          approvedThisMonth: approved,
          totalWorkload: pending + approved,
          approvalRate: pending + approved > 0 ? 
            Math.round((approved / (pending + approved)) * 100) : 0
        };
      })
    );
  }

  /**
   * Generate program statistics report
   */
  async generateProgramStatistics(query) {
    const programs = await Apprentice.distinct('program');

    return Promise.all(
      programs.map(async (program) => {
        const apprentices = await Apprentice.find({ program });
        const hoursData = await HourLog.find({
          apprenticeId: { $in: apprentices.map(a => a._id) },
          status: 'approved',
          ...query
        });

        const totalHours = hoursData.reduce((sum, log) => {
          return sum + (log.qualifiedHours || 0);
        }, 0);

        const avgHours = apprentices.length > 0 ? 
          Math.round(totalHours / apprentices.length) : 0;

        return {
          program,
          totalApprentices: apprentices.length,
          totalHours,
          averageHoursPerApprentice: avgHours,
          completionRate: apprentices.filter(a => {
            const prog = a.status === 'completed' ? 1 : 0;
            return prog;
          }).length
        };
      })
    );
  }

  /**
   * Generate completion forecast
   */
  async generateCompletionForecast(query) {
    const apprentices = await Apprentice.find(query).lean();

    return apprentices.map(apprentice => {
      // Placeholder forecast logic
      const estimatedCompletion = new Date();
      estimatedCompletion.setMonth(estimatedCompletion.getMonth() + 12);

      return {
        apprenticeName: `${apprentice.firstName} ${apprentice.lastName}`,
        program: apprentice.program,
        estimatedCompletion: estimatedCompletion.toISOString().split('T')[0],
        confidence: 'High'
      };
    });
  }

  /**
   * Generate custom report
   */
  async generateCustomReport(query, columns) {
    const logs = await HourLog.find(query)
      .populate('apprenticeId', 'firstName lastName')
      .lean();

    return logs.map(log => {
      const row = {};
      columns.forEach(col => {
        row[col.name] = log[col.name] || '';
      });
      return row;
    });
  }

  /**
   * Group data by field
   */
  groupData(data, field) {
    const grouped = {};

    data.forEach(item => {
      const key = item[field] || 'Other';
      if (!grouped[key]) {
        grouped[key] = [];
      }
      grouped[key].push(item);
    });

    return Object.entries(grouped).map(([key, items]) => ({
      [field]: key,
      items,
      count: items.length
    }));
  }

  /**
   * Sort data
   */
  sortData(data, field, order) {
    const sorted = [...data];
    sorted.sort((a, b) => {
      if (a[field] < b[field]) return order === 'asc' ? -1 : 1;
      if (a[field] > b[field]) return order === 'asc' ? 1 : -1;
      return 0;
    });
    return sorted;
  }

  /**
   * Update report
   */
  async updateReport(reportId, userId, updateData) {
    try {
      const report = await Report.findById(reportId);

      if (!report || (report.owner.toString() !== userId && report.createdBy.toString() !== userId)) {
        throw new Error('Unauthorized');
      }

      Object.assign(report, updateData);
      await report.save();

      return report;
    } catch (error) {
      console.error('Error updating report:', error);
      throw error;
    }
  }

  /**
   * Delete report
   */
  async deleteReport(reportId, userId) {
    try {
      const report = await Report.findById(reportId);

      if (!report || report.owner.toString() !== userId) {
        throw new Error('Unauthorized');
      }

      await Report.deleteOne({ _id: reportId });
      return true;
    } catch (error) {
      console.error('Error deleting report:', error);
      throw error;
    }
  }

  /**
   * Share report with user
   */
  async shareReport(reportId, userId, targetUserId, permission = 'view') {
    try {
      const report = await Report.findById(reportId);

      if (!report || report.owner.toString() !== userId) {
        throw new Error('Unauthorized');
      }

      const existing = report.sharedWith.findIndex(s => s.userId.toString() === targetUserId);
      if (existing >= 0) {
        report.sharedWith[existing].permission = permission;
      } else {
        report.sharedWith.push({ userId: targetUserId, permission });
      }

      await report.save();
      return report;
    } catch (error) {
      console.error('Error sharing report:', error);
      throw error;
    }
  }

  /**
   * Get scheduled reports due for execution
   */
  async getScheduledReportsForExecution() {
    try {
      const now = new Date();
      return await Report.find({
        'schedule.enabled': true,
        'schedule.nextRun': { $lte: now },
        status: 'active'
      });
    } catch (error) {
      console.error('Error fetching scheduled reports:', error);
      throw error;
    }
  }

  /**
   * Execute scheduled report
   */
  async executeScheduledReport(reportId) {
    try {
      const report = await Report.findById(reportId);
      if (!report) throw new Error('Report not found');

      const reportData = await this.generateReportData(report);

      // Log execution
      report.executions.push({
        executedAt: new Date(),
        executedBy: report.owner,
        rowCount: reportData.rowCount
      });

      report.lastGeneratedAt = new Date();
      report.lastRowCount = reportData.rowCount;

      // Recalculate next run
      if (report.schedule.frequency !== 'once') {
        report.schedule.nextRun = calculateNextRun(report.schedule);
      } else {
        report.schedule.enabled = false;
      }

      await report.save();

      return reportData;
    } catch (error) {
      console.error('Error executing scheduled report:', error);
      throw error;
    }
  }
}

export default new ReportService();
