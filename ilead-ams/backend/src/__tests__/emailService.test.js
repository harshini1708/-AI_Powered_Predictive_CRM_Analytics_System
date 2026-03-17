import emailService from '../services/emailService.js';

// Mock nodemailer
jest.mock('nodemailer');

describe('Email Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    emailService.transporter = null;
  });

  describe('Initialize', () => {
    it('should initialize transporter with SMTP config', () => {
      process.env.SMTP_USER = 'test@gmail.com';
      process.env.SMTP_PASS = 'password123';
      
      emailService.initialize();
      
      expect(emailService.transporter).toBeDefined();
    });

    it('should skip initialization if SMTP credentials missing', () => {
      process.env.SMTP_USER = '';
      process.env.SMTP_PASS = '';
      
      emailService.initialize();
      
      expect(emailService.transporter).toBeNull();
    });
  });

  describe('Send Email', () => {
    beforeEach(() => {
      emailService.transporter = {
        sendMail: jest.fn().mockResolvedValue({ messageId: 'test-id' })
      };
    });

    it('should send email with correct parameters', async () => {
      const result = await emailService.sendEmail(
        'user@example.com',
        'Test Subject',
        '<html>Test</html>'
      );

      expect(result).toBe(true);
      expect(emailService.transporter.sendMail).toHaveBeenCalledWith(
        expect.objectContaining({
          to: 'user@example.com',
          subject: 'Test Subject',
          html: '<html>Test</html>'
        })
      );
    });

    it('should return false if transporter not initialized', async () => {
      emailService.transporter = null;
      
      const result = await emailService.sendEmail('user@example.com', 'Subject', '<html></html>');
      
      expect(result).toBe(false);
    });

    it('should handle send errors gracefully', async () => {
      emailService.transporter.sendMail.mockRejectedValue(new Error('SMTP Error'));
      
      const result = await emailService.sendEmail('user@example.com', 'Subject', '<html></html>');
      
      expect(result).toBe(false);
    });
  });

  describe('Hour Submission Confirmation', () => {
    beforeEach(() => {
      emailService.transporter = {
        sendMail: jest.fn().mockResolvedValue({ messageId: 'test-id' })
      };
    });

    it('should send hour submission confirmation email', async () => {
      const user = { email: 'apprentice@example.com', firstName: 'John' };
      const hourLog = { date: new Date('2026-03-17'), ojtHours: 8 };

      const result = await emailService.sendHourSubmissionConfirmation(user, hourLog);

      expect(result).toBe(true);
      expect(emailService.transporter.sendMail).toHaveBeenCalled();
      
      const call = emailService.transporter.sendMail.mock.calls[0][0];
      expect(call.to).toBe('apprentice@example.com');
      expect(call.subject).toContain('Submission Confirmation');
    });
  });

  describe('Hour Approval', () => {
    beforeEach(() => {
      emailService.transporter = {
        sendMail: jest.fn().mockResolvedValue({ messageId: 'test-id' })
      };
    });

    it('should send hour approval notification', async () => {
      const apprentice = { email: 'apprentice@example.com', firstName: 'John' };
      const hourLog = { date: new Date('2026-03-17'), qualifiedHours: 8 };
      const supervisor = { firstName: 'Jane', lastName: 'Doe' };

      const result = await emailService.sendHourApprovalNotification(apprentice, hourLog, supervisor);

      expect(result).toBe(true);
      expect(emailService.transporter.sendMail).toHaveBeenCalled();
      
      const call = emailService.transporter.sendMail.mock.calls[0][0];
      expect(call.to).toBe('apprentice@example.com');
      expect(call.subject).toContain('Approved');
    });
  });

  describe('Hour Rejection', () => {
    beforeEach(() => {
      emailService.transporter = {
        sendMail: jest.fn().mockResolvedValue({ messageId: 'test-id' })
      };
    });

    it('should send hour rejection notification', async () => {
      const apprentice = { email: 'apprentice@example.com', firstName: 'John' };
      const hourLog = { date: new Date('2026-03-17') };
      const supervisor = { firstName: 'Jane' };
      const reason = 'Missing documentation';

      const result = await emailService.sendHourRejectionNotification(
        apprentice,
        hourLog,
        supervisor,
        reason
      );

      expect(result).toBe(true);
      
      const call = emailService.transporter.sendMail.mock.calls[0][0];
      expect(call.subject).toContain('Revision');
    });
  });

  describe('Batch Email', () => {
    beforeEach(() => {
      emailService.transporter = {
        sendMail: jest.fn().mockResolvedValue({ messageId: 'test-id' })
      };
    });

    it('should send batch emails to multiple recipients', async () => {
      const recipients = ['user1@example.com', 'user2@example.com', 'user3@example.com'];
      const subject = 'Test Subject';
      const html = '<html>Test</html>';

      const results = await emailService.sendBatchEmail(recipients, subject, html);

      expect(results).toHaveLength(3);
      expect(results.every(r => r.success)).toBe(true);
      expect(emailService.transporter.sendMail).toHaveBeenCalledTimes(3);
    });

    it('should handle partial failures in batch sending', async () => {
      emailService.transporter.sendMail
        .mockResolvedValueOnce({ messageId: 'test-id' })
        .mockRejectedValueOnce(new Error('Failed'))
        .mockResolvedValueOnce({ messageId: 'test-id' });

      const recipients = ['user1@example.com', 'user2@example.com', 'user3@example.com'];
      const results = await emailService.sendBatchEmail(recipients, 'Subject', '<html></html>');

      expect(results).toHaveLength(3);
      expect(results[0].success).toBe(true);
      expect(results[1].success).toBe(false);
      expect(results[2].success).toBe(true);
    });
  });

  describe('Verify Connection', () => {
    it('should verify SMTP connection', async () => {
      emailService.transporter = {
        verify: jest.fn().mockResolvedValue(true)
      };

      const status = await emailService.verifyConnection();

      expect(status.status).toBe('connected');
      expect(emailService.transporter.verify).toHaveBeenCalled();
    });

    it('should report error on connection failure', async () => {
      emailService.transporter = {
        verify: jest.fn().mockRejectedValue(new Error('SMTP error'))
      };

      const status = await emailService.verifyConnection();

      expect(status.status).toBe('error');
      expect(status.message).toContain('SMTP error');
    });

    it('should handle missing transporter', async () => {
      emailService.transporter = null;

      const status = await emailService.verifyConnection();

      expect(status.status).toBe('disabled');
    });
  });
});
