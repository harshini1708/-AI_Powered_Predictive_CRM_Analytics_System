const db = require('../config/database');

module.exports = {
  handleMakeWebhook: async (req, res) => {
    try {
      const { action, ...payload } = req.body;

      console.log(`Webhook received: ${action}`, payload);

      switch (action) {
        case 'registration':
          // Log registration to database for audit trail
          await db('audit_logs').insert({
            id: require('uuid').v4(),
            user_id: null,
            action: 'registration_webhook',
            resource_type: 'user',
            details: JSON.stringify(payload),
            created_at: new Date(),
            updated_at: new Date()
          });
          break;

        case 'new_apprentice':
          await db('audit_logs').insert({
            id: require('uuid').v4(),
            user_id: null,
            action: 'new_apprentice_webhook',
            resource_type: 'apprentice',
            details: JSON.stringify(payload),
            created_at: new Date(),
            updated_at: new Date()
          });
          break;

        case 'log_hours':
          await db('audit_logs').insert({
            id: require('uuid').v4(),
            user_id: null,
            action: 'log_hours_webhook',
            resource_type: 'hour_log',
            details: JSON.stringify(payload),
            created_at: new Date(),
            updated_at: new Date()
          });
          break;

        default:
          console.log(`Unknown webhook action: ${action}`);
      }

      res.json({
        message: 'Webhook received and processed',
        action: action,
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Webhook error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  getStatus: async (req, res) => {
    res.json({
      status: 'operational',
      message: 'Webhook endpoint is ready to receive events',
      timestamp: new Date().toISOString()
    });
  }
};
