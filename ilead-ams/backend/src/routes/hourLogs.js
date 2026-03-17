const express = require('express');
const router = express.Router();
const Joi = require('joi');
const { authenticate } = require('../middleware/auth');
const { isReviewerRole, roleCheck } = require('../middleware/roleCheck');
const { validate } = require('../middleware/validation');
const hourLogController = require('../controllers/hourLogController');

const submitHoursSchema = Joi.object({
  ojtDomain: Joi.string().valid('dataCollection', 'assessment', 'skillAcquisition', 'behaviorReduction', 'documentation', 'crisisManagement'),
  rtiModule: Joi.string().valid('ethics', 'measurement', 'assessProc', 'skillStrat', 'behavTech', 'docStandards', 'crisis', 'commSkills', 'legal', 'cultural', 'tech', 'profDev', 'capstone'),
  ojtHours: Joi.number().min(0).max(12),
  rtiHours: Joi.number().min(0).max(12),
  logDate: Joi.string().isoDate().required(),
  description: Joi.string().required()
});

const approveHoursSchema = Joi.object({
  rubricScore: Joi.number().min(1).max(5).required(),
  rubricDomain: Joi.string().valid('A', 'B', 'C', 'D', 'E', 'F').required(),
  specificTask: Joi.string().required(),
  rubricNotes: Joi.string().required(),
  nextSteps: Joi.string().allow(''),
  remediationRequired: Joi.boolean(),
  supervisionMinutes: Joi.number().min(0),
  supervisionType: Joi.string().valid('direct', 'indirect').required(),
  accuracyData: Joi.string().allow(''),
  contextVars: Joi.string().allow('')
});

const rejectHoursSchema = Joi.object({
  rubricNotes: Joi.string().required()
});

router.post('/', authenticate, roleCheck('apprentice'), validate(submitHoursSchema), hourLogController.submit);
router.get('/', authenticate, hourLogController.list);
router.get('/:id', authenticate, hourLogController.getById);
router.put('/:id/approve', authenticate, isReviewerRole, validate(approveHoursSchema), hourLogController.approve);
router.put('/:id/reject', authenticate, isReviewerRole, validate(rejectHoursSchema), hourLogController.reject);

module.exports = router;
