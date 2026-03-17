const axios = require('axios');

const CLEARSTREAM_API_KEY = process.env.CLEARSTREAM_API_KEY;
const CLEARSTREAM_ENDPOINT = process.env.CLEARSTREAM_ENDPOINT || 'https://api.getclearstream.com/v1/texts';

// SMS message templates
const templates = {
  accountApproved: (name) => `Hi ${name}, your I-LEAD AMS account has been approved! You can now log in.`,
  hourApproved: (apprenticeName, hours) => `Hi ${apprenticeName}, your ${hours}h submission was approved!`,
  hourRejected: (apprenticeName) => `Hi ${apprenticeName}, your hour submission needs revision. Please log in to review feedback.`,
  pendingApproval: (name) => `Hi ${name}, your I-LEAD AMS account is pending approval. You'll be notified when it's ready.`,
  reminderPendingHours: (supervisorName) => `Hi ${supervisorName}, you have pending hour submissions to review in I-LEAD AMS.`
};

class SMSService {
  constructor() {
    this.apiKey = CLEARSTREAM_API_KEY;
    this.endpoint = CLEARSTREAM_ENDPOINT;
    this.isDevelopment = !CLEARSTREAM_API_KEY;
  }

  async send(phoneNumber, templateName, ...templateData) {
    try {
      if (!templates[templateName]) {
        throw new Error(`SMS template "${templateName}" not found`);
      }

      const message = templates[templateName](...templateData);

      // Development mode: log instead of send
      if (this.isDevelopment) {
        console.log(`\n📱 SMS (Development Mode):`);
        console.log(`   To: ${phoneNumber}`);
        console.log(`   Message: ${message}`);
        return { success: true, messageId: 'dev-sms-' + Date.now() };
      }

      if (!this.apiKey) {
        console.warn('Clearstream API key not configured, skipping SMS send');
        return { success: false, messageId: null };
      }

      // Normalize phone number (remove dashes, spaces, etc.)
      const cleanPhone = phoneNumber.replace(/\D/g, '');
      if (cleanPhone.length < 10) {
        console.warn(`Invalid phone number format: ${phoneNumber}`);
        return { success: false, error: 'Invalid phone number' };
      }

      const response = await axios.post(
        this.endpoint,
        {
          to: [cleanPhone],
          text_body: message
        },
        {
          headers: {
            'X-Api-Key': this.apiKey,
            'Content-Type': 'application/json'
          },
          timeout: 10000 // 10 second timeout
        }
      );

      console.log(`✅ SMS sent to ${phoneNumber}: ${response.data.messageId || 'success'}`);
      return { success: true, messageId: response.data.messageId };
    } catch (err) {
      console.error(`❌ SMS send failed for ${phoneNumber}:`, err.message);
      // Don't throw - SMS failures shouldn't break the app
      return { success: false, error: err.message };
    }
  }

  // Specific SMS methods
  async sendAccountApprovedSMS(phone, name) {
    return this.send(phone, 'accountApproved', name);
  }

  async sendHourApprovedSMS(phone, apprenticeName, hours) {
    return this.send(phone, 'hourApproved', apprenticeName, hours);
  }

  async sendHourRejectedSMS(phone, apprenticeName) {
    return this.send(phone, 'hourRejected', apprenticeName);
  }

  async sendPendingApprovalSMS(phone, name) {
    return this.send(phone, 'pendingApproval', name);
  }

  async sendReminderPendingHoursSMS(phone, supervisorName) {
    return this.send(phone, 'reminderPendingHours', supervisorName);
  }

  // Batch send (for notifications to multiple supervisors)
  async sendBatch(recipients, templateName, ...templateData) {
    const results = [];
    for (const phone of recipients) {
      const result = await this.send(phone, templateName, ...templateData);
      results.push({ phone, ...result });
    }
    return results;
  }
}

module.exports = new SMSService();
