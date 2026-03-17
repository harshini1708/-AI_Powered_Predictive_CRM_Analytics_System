/**
 * Email Templates for I-LEAD AMS
 */

/**
 * Hour submission confirmation template
 */
export const hourSubmissionConfirmation = (user, hourLog) => `
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
    .header { background: #003d82; color: white; padding: 20px; border-radius: 5px; text-align: center; }
    .content { padding: 20px; background: #f8f9fa; margin-top: 20px; border-radius: 5px; }
    .details { background: white; padding: 15px; margin: 10px 0; border-left: 4px solid #e84c1f; }
    .footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
    .button { background: #003d82; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Hour Log Submission Confirmation</h1>
    </div>
    
    <div class="content">
      <p>Hello ${user.firstName},</p>
      
      <p>Your hour log has been successfully submitted! Here are the details:</p>
      
      <div class="details">
        <strong>Date:</strong> ${new Date(hourLog.date).toLocaleDateString()}<br>
        <strong>OJT Hours:</strong> ${hourLog.ojtHours || 0}<br>
        <strong>RTI Hours:</strong> ${hourLog.rtiHours || 0}<br>
        <strong>Domain:</strong> ${hourLog.ojtDomainName || 'N/A'}<br>
        <strong>Status:</strong> Pending Approval
      </div>
      
      <p>Your submission will be reviewed by your supervisor shortly. You'll receive another notification once it's been approved or if revisions are needed.</p>
      
      <p>
        <a href="${process.env.FRONTEND_URL}/hours" class="button">View My Hours</a>
      </p>
      
      <p>Questions? Contact your supervisor or the administration team.</p>
      
      <p>Best regards,<br>I-LEAD AMS Team</p>
    </div>
    
    <div class="footer">
      <p>© ${new Date().getFullYear()} I-LEAD AMS. All rights reserved.</p>
    </div>
  </div>
</body>
</html>
`;

/**
 * Hour approval notification template
 */
export const hourApprovalNotification = (user, hourLog, supervisor) => `
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
    .header { background: #28a745; color: white; padding: 20px; border-radius: 5px; text-align: center; }
    .content { padding: 20px; background: #f8f9fa; margin-top: 20px; border-radius: 5px; }
    .details { background: white; padding: 15px; margin: 10px 0; border-left: 4px solid #28a745; }
    .footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>✅ Hour Log Approved</h1>
    </div>
    
    <div class="content">
      <p>Hello ${user.firstName},</p>
      
      <p>Great news! Your hour log has been approved by ${supervisor?.firstName || 'your supervisor'}.</p>
      
      <div class="details">
        <strong>Date:</strong> ${new Date(hourLog.date).toLocaleDateString()}<br>
        <strong>Hours Approved:</strong> ${hourLog.qualifiedHours || 0}<br>
        <strong>Approved By:</strong> ${supervisor?.firstName} ${supervisor?.lastName}<br>
        <strong>Status:</strong> ✅ Approved
      </div>
      
      <p>These hours have been added to your progress record. Keep up the great work!</p>
      
      <p>Best regards,<br>I-LEAD AMS Team</p>
    </div>
    
    <div class="footer">
      <p>© ${new Date().getFullYear()} I-LEAD AMS. All rights reserved.</p>
    </div>
  </div>
</body>
</html>
`;

/**
 * Hour rejection notification template
 */
export const hourRejectionNotification = (user, hourLog, supervisor, reason) => `
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
    .header { background: #dc3545; color: white; padding: 20px; border-radius: 5px; text-align: center; }
    .content { padding: 20px; background: #f8f9fa; margin-top: 20px; border-radius: 5px; }
    .details { background: white; padding: 15px; margin: 10px 0; border-left: 4px solid #dc3545; }
    .footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>⚠️ Hour Log Requires Revision</h1>
    </div>
    
    <div class="content">
      <p>Hello ${user.firstName},</p>
      
      <p>Your hour log submission requires revision. Please review the feedback below and resubmit.</p>
      
      <div class="details">
        <strong>Date:</strong> ${new Date(hourLog.date).toLocaleDateString()}<br>
        <strong>Reason:</strong> ${reason || 'Please contact your supervisor for details'}<br>
        <strong>Reviewed By:</strong> ${supervisor?.firstName} ${supervisor?.lastName}
      </div>
      
      <p>Please make the necessary corrections and resubmit your hour log. If you have any questions, please reach out to your supervisor.</p>
      
      <p>Best regards,<br>I-LEAD AMS Team</p>
    </div>
    
    <div class="footer">
      <p>© ${new Date().getFullYear()} I-LEAD AMS. All rights reserved.</p>
    </div>
  </div>
</body>
</html>
`;

/**
 * Pending approvals digest template
 */
export const pendingApprovalsDigest = (supervisor, pendingLogs) => `
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
    .header { background: #003d82; color: white; padding: 20px; border-radius: 5px; text-align: center; }
    .content { padding: 20px; background: #f8f9fa; margin-top: 20px; border-radius: 5px; }
    table { width: 100%; border-collapse: collapse; margin: 15px 0; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #e84c1f; color: white; }
    .footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
    .button { background: #003d82; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>⏰ Pending Approvals Summary</h1>
    </div>
    
    <div class="content">
      <p>Hello ${supervisor.firstName},</p>
      
      <p>You have ${pendingLogs.length} pending hour log approval(s) waiting for your review:</p>
      
      <table>
        <thead>
          <tr>
            <th>Apprentice</th>
            <th>Date</th>
            <th>Hours</th>
            <th>Submitted</th>
          </tr>
        </thead>
        <tbody>
          ${pendingLogs.map(log => `
            <tr>
              <td>${log.apprenticeName || log.apprenticeId}</td>
              <td>${new Date(log.date).toLocaleDateString()}</td>
              <td>${log.ojtHours + (log.rtiHours || 0)}</td>
              <td>${new Date(log.createdAt).toLocaleDateString()}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
      
      <p>
        <a href="${process.env.FRONTEND_URL}/approvals" class="button">Review Approvals</a>
      </p>
      
      <p>Best regards,<br>I-LEAD AMS Team</p>
    </div>
    
    <div class="footer">
      <p>© ${new Date().getFullYear()} I-LEAD AMS. All rights reserved.</p>
    </div>
  </div>
</body>
</html>
`;

/**
 * Account created notification template
 */
export const accountCreatedNotification = (user, tempPassword) => `
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
    .header { background: #003d82; color: white; padding: 20px; border-radius: 5px; text-align: center; }
    .content { padding: 20px; background: #f8f9fa; margin-top: 20px; border-radius: 5px; }
    .credentials { background: white; padding: 15px; margin: 15px 0; border: 1px solid #ddd; border-radius: 5px; font-family: monospace; }
    .footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
    .button { background: #003d82; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0; }
    .warning { background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Welcome to I-LEAD AMS!</h1>
    </div>
    
    <div class="content">
      <p>Hello ${user.firstName},</p>
      
      <p>Your I-LEAD AMS account has been created! You can now log in with your credentials.</p>
      
      <div class="credentials">
        <strong>Email:</strong> ${user.email}<br>
        <strong>Temporary Password:</strong> ${tempPassword}
      </div>
      
      <div class="warning">
        <strong>⚠️ Important:</strong> For security, please change your password immediately after your first login.
      </div>
      
      <p>
        <a href="${process.env.FRONTEND_URL}/login" class="button">Log In Now</a>
      </p>
      
      <p>If you have any questions or need assistance, please contact the administration team.</p>
      
      <p>Best regards,<br>I-LEAD AMS Team</p>
    </div>
    
    <div class="footer">
      <p>© ${new Date().getFullYear()} I-LEAD AMS. All rights reserved.</p>
    </div>
  </div>
</body>
</html>
`;

/**
 * Password reset email template
 */
export const passwordResetEmail = (user, resetUrl) => `
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
    .header { background: #003d82; color: white; padding: 20px; border-radius: 5px; text-align: center; }
    .content { padding: 20px; background: #f8f9fa; margin-top: 20px; border-radius: 5px; }
    .footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
    .button { background: #003d82; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0; }
    .warning { background: #f8d7da; padding: 15px; border-left: 4px solid #dc3545; margin: 15px 0; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Password Reset Request</h1>
    </div>
    
    <div class="content">
      <p>Hello ${user.firstName},</p>
      
      <p>We received a request to reset your password. Click the button below to set a new password:</p>
      
      <p>
        <a href="${resetUrl}" class="button">Reset Password</a>
      </p>
      
      <p>Or copy and paste this link into your browser:<br>
      <code>${resetUrl}</code></p>
      
      <div class="warning">
        <strong>⚠️ Security Note:</strong> This link expires in 1 hour. If you didn't request this, please ignore this email.
      </div>
      
      <p>Best regards,<br>I-LEAD AMS Team</p>
    </div>
    
    <div class="footer">
      <p>© ${new Date().getFullYear()} I-LEAD AMS. All rights reserved.</p>
    </div>
  </div>
</body>
</html>
`;

/**
 * Weekly summary template
 */
export const weeklySummary = (user, summary) => `
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
    .header { background: #003d82; color: white; padding: 20px; border-radius: 5px; text-align: center; }
    .content { padding: 20px; background: #f8f9fa; margin-top: 20px; border-radius: 5px; }
    .stat-box { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; text-align: center; }
    .stat-number { font-size: 24px; font-weight: bold; color: #003d82; }
    .stat-label { color: #666; font-size: 14px; }
    .footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
    .button { background: #003d82; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 0; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>📊 Your Weekly Summary</h1>
    </div>
    
    <div class="content">
      <p>Hello ${user.firstName},</p>
      
      <p>Here's your progress for this week:</p>
      
      <div class="stat-box">
        <div class="stat-number">${summary.hoursLogged}</div>
        <div class="stat-label">Total Hours Logged</div>
      </div>
      
      <div class="stat-box">
        <div class="stat-number">${summary.hoursApproved}</div>
        <div class="stat-label">Hours Approved</div>
      </div>
      
      <div class="stat-box">
        <div class="stat-number">${summary.progressPercentage}%</div>
        <div class="stat-label">Progress Toward Goal</div>
      </div>
      
      <p>Keep up the great work! You're making excellent progress in your apprenticeship.</p>
      
      <p>
        <a href="${process.env.FRONTEND_URL}/dashboard" class="button">View Dashboard</a>
      </p>
      
      <p>Best regards,<br>I-LEAD AMS Team</p>
    </div>
    
    <div class="footer">
      <p>© ${new Date().getFullYear()} I-LEAD AMS. All rights reserved.</p>
    </div>
  </div>
</body>
</html>
`;

/**
 * Admin alert template
 */
export const adminAlert = (alertType, details) => `
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
    .container { max-width: 600px; margin: 0 auto; padding: 20px; }
    .header { background: #dc3545; color: white; padding: 20px; border-radius: 5px; text-align: center; }
    .content { padding: 20px; background: #f8f9fa; margin-top: 20px; border-radius: 5px; }
    .details { background: white; padding: 15px; margin: 10px 0; border-left: 4px solid #dc3545; font-family: monospace; }
    .footer { text-align: center; padding: 20px; font-size: 12px; color: #666; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🚨 System Alert</h1>
    </div>
    
    <div class="content">
      <p><strong>Alert Type:</strong> ${alertType}</p>
      
      <div class="details">
        ${Object.entries(details || {}).map(([key, value]) => 
          `<strong>${key}:</strong> ${value}<br>`
        ).join('')}
      </div>
      
      <p>Please investigate this issue and take necessary action.</p>
      
      <p>Best regards,<br>I-LEAD AMS System</p>
    </div>
    
    <div class="footer">
      <p>© ${new Date().getFullYear()} I-LEAD AMS. All rights reserved.</p>
    </div>
  </div>
</body>
</html>
`;
