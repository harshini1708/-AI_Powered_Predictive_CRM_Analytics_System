const express = require('express');
const router = express.Router();
const { authenticate } = require('../middleware/auth');
const analyticsController = require('../controllers/analyticsController');

router.get('/dashboard', authenticate, analyticsController.dashboard);
router.get('/domain-progress', authenticate, analyticsController.domainProgress);
router.get('/competency-heat-map', authenticate, analyticsController.competencyHeatMap);

module.exports = router;
