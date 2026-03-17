const express = require('express');
const router = express.Router();
const webhookController = require('../controllers/webhookController');

// Make.com webhook - no authentication required (Make.com sends POST directly)
router.post('/make', webhookController.handleMakeWebhook);

// Health check
router.get('/status', webhookController.getStatus);

module.exports = router;
