import cron from 'node-cron';
import reportService from './reportService.js';
import emailService from './emailService.js';
import Report from '../models/Report.js';

/**
 * Report Scheduler - Handles scheduled report execution
 */
class ReportScheduler {
  constructor() {
    this.jobs = new Map();
    this.isRunning = false;
  }

  /**
   * Start scheduler
   */
  start() {
    if (this.isRunning) {
      console.log('📋 Report scheduler already running');
      return;
    }

    console.log('📋 Starting report scheduler...');

    // Check for scheduled reports every minute
    this.jobs.set('scheduler-check', cron.schedule('* * * * *', async () => {
      await this.checkAndExecuteScheduledReports();
    }));

    this.isRunning = true;
    console.log('✅ Report scheduler started');
  }

  /**
   * Stop scheduler
   */
  stop() {
    console.log('📋 Stopping report scheduler...');
    
    this.jobs.forEach((job, key) => {
      job.stop();
      this.jobs.delete(key);
    });

    this.isRunning = false;
    console.log('✅ Report scheduler stopped');
  }

  /**
   * Check and execute scheduled reports
   */
  async checkAndExecuteScheduledReports() {
    try {
      const reports = await reportService.getScheduledReportsForExecution();

      if (reports.length === 0) return;

      console.log(`⏰ Found ${reports.length} scheduled reports to execute`);

      for (const report of reports) {
        await this.executeScheduledReport(report);
      }
    } catch (error) {
      console.error('Error checking scheduled reports:', error);
    }
  }

  /**
   * Execute scheduled report
   */
  async executeScheduledReport(report) {
    try {
      console.log(`⏳ Executing scheduled report: ${report.name}`);

      const reportData = await reportService.executeScheduledReport(report._id);

      // Send email if delivery enabled
      if (report.delivery?.enabled && report.delivery?.recipients?.length > 0) {
        await this.deliverReport(report, reportData);
      }

      console.log(`✅ Report executed: ${report.name} (${reportData.rowCount} rows)`);
    } catch (error) {
      console.error(`Error executing scheduled report ${report.name}:`, error);

      // Log error in execution history
      try {
        await Report.findByIdAndUpdate(report._id, {
          $push: {
            executions: {
              executedAt: new Date(),
              executedBy: report.owner,
              error: error.message
            }
          }
        });
      } catch (updateError) {
        console.error('Error logging execution error:', updateError);
      }
    }
  }

  /**
   * Deliver report via email
   */
  async deliverReport(report, reportData) {
    try {
      const recipients = report.delivery.recipients;
      const format = report.delivery.format || 'pdf';

      // Generate email content
      const emailContent = this.generateReportEmail(report, reportData);

      // Send to each recipient
      for (const recipient of recipients) {
        try {
          await emailService.sendEmail(
            recipient,
            `Scheduled Report: ${report.name}`,
            emailContent
          );
          console.log(`📧 Report sent to ${recipient}`);
        } catch (error) {
          console.error(`Error sending report to ${recipient}:`, error);
        }
      }
    } catch (error) {
      console.error('Error delivering report:', error);
    }
  }

  /**
   * Generate report email HTML
   */
  generateReportEmail(report, reportData) {
    const rowsPreview = reportData.data.slice(0, 10).map(row => {
      return `<tr>${Object.values(row).map(v => `<td>${v}</td>`).join('')}</tr>`;
    }).join('');

    return `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: Arial, sans-serif; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: #003d82; color: white; padding: 20px; border-radius: 5px; }
          .content { margin: 20px 0; }
          table { width: 100%; border-collapse: collapse; margin: 15px 0; }
          th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
          th { background: #f8f9fa; font-weight: bold; }
          .footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>${report.name}</h1>
            <p>Scheduled Report - Generated on ${new Date().toLocaleString()}</p>
          </div>

          <div class="content">
            <p>${report.description || ''}</p>

            <h3>Summary</h3>
            <p>Total Records: ${reportData.rowCount}</p>
            <p>Generated: ${new Date(reportData.generatedAt).toLocaleString()}</p>

            <h3>Data Preview (first 10 rows)</h3>
            <table>
              <thead>
                <tr>
                  ${Object.keys(reportData.data[0] || {}).map(k => `<th>${k}</th>`).join('')}
                </tr>
              </thead>
              <tbody>
                ${rowsPreview}
              </tbody>
            </table>

            <p style="text-align: center;">
              <a href="${process.env.FRONTEND_URL}/reports/${report._id}" 
                 style="background: #003d82; color: white; padding: 10px 20px; 
                        text-decoration: none; border-radius: 5px;">
                View Full Report
              </a>
            </p>
          </div>

          <div class="footer">
            <p>This is an automated report. Please do not reply to this email.</p>
            <p>&copy; ${new Date().getFullYear()} I-LEAD AMS</p>
          </div>
        </div>
      </body>
      </html>
    `;
  }

  /**
   * Reschedule report
   */
  async rescheduleReport(reportId) {
    try {
      const report = await Report.findById(reportId);
      if (!report || !report.schedule.enabled) {
        return;
      }

      // Check if job already exists
      if (this.jobs.has(reportId)) {
        this.jobs.get(reportId).stop();
        this.jobs.delete(reportId);
      }

      // Calculate cron expression
      const cronExpression = this.getCronExpression(report.schedule);
      if (!cronExpression) return;

      // Create new scheduled job
      const job = cron.schedule(cronExpression, async () => {
        await this.executeScheduledReport(report);
      });

      this.jobs.set(reportId, job);
      console.log(`✅ Scheduled report: ${report.name}`);
    } catch (error) {
      console.error('Error rescheduling report:', error);
    }
  }

  /**
   * Convert schedule to cron expression
   */
  getCronExpression(schedule) {
    const { frequency, time = '09:00', dayOfWeek = 1, dayOfMonth = 1 } = schedule;
    const [hours, minutes] = time.split(':');

    switch (frequency) {
      case 'daily':
        return `${minutes} ${hours} * * *`;
      case 'weekly':
        return `${minutes} ${hours} * * ${dayOfWeek}`;
      case 'monthly':
        return `${minutes} ${hours} ${dayOfMonth} * *`;
      case 'quarterly':
        return `${minutes} ${hours} ${dayOfMonth} */3 *`;
      case 'yearly':
        return `${minutes} ${hours} ${dayOfMonth} 1 *`;
      default:
        return null;
    }
  }

  /**
   * Get scheduler status
   */
  getStatus() {
    return {
      running: this.isRunning,
      totalJobs: this.jobs.size,
      jobs: Array.from(this.jobs.keys())
    };
  }
}

export default new ReportScheduler();
