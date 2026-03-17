const express = require('express');
const router = express.Router();
const Joi = require('joi');
const { authenticate } = require('../middleware/auth');
const { isSuperAdmin } = require('../middleware/roleCheck');
const { validate } = require('../middleware/validation');
const programController = require('../controllers/programController');

const createProgramSchema = Joi.object({
  code: Joi.string().required(),
  name: Joi.string().required(),
  type: Joi.string().valid('registeredApprenticeship', 'preApprenticeship', 'workBasedLearning', 'industryCertification'),
  targetOjtHours: Joi.number().required(),
  targetRtiHours: Joi.number().required(),
  partner: Joi.string().allow('')
});

router.get('/', authenticate, programController.list);
router.get('/:id', authenticate, programController.getById);
router.post('/', authenticate, isSuperAdmin, validate(createProgramSchema), programController.create);
router.delete('/:id', authenticate, isSuperAdmin, programController.delete);

module.exports = router;
