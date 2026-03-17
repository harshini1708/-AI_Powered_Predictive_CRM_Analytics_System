const express = require('express');
const router = express.Router();
const Joi = require('joi');
const { authenticate } = require('../middleware/auth');
const { isAdmin } = require('../middleware/roleCheck');
const { validate } = require('../middleware/validation');
const apprenticeController = require('../controllers/apprenticeController');

const createApprenticeSchema = Joi.object({
  email: Joi.string().email().required().lowercase().trim(),
  firstName: Joi.string().required().trim(),
  lastName: Joi.string().required().trim(),
  phone: Joi.string().allow('').trim(),
  programCode: Joi.string().required(),
  supervisor: Joi.string().allow('').trim(),
  supervisorEmail: Joi.string().email().allow('').lowercase().trim(),
  journeyworker: Joi.string().allow('').trim(),
  journeyworkerEmail: Joi.string().email().allow('').lowercase().trim(),
  employer: Joi.string().allow('').trim(),
  startDate: Joi.string().isoDate().required(),
  wageStart: Joi.number().allow(null),
  wageCurrent: Joi.number().allow(null)
});

const updateApprenticeSchema = Joi.object({
  supervisor: Joi.string().allow('').trim(),
  supervisorEmail: Joi.string().email().allow('').lowercase().trim(),
  journeyworker: Joi.string().allow('').trim(),
  journeyworkerEmail: Joi.string().email().allow('').lowercase().trim(),
  employer: Joi.string().allow('').trim(),
  wageCurrent: Joi.number(),
  status: Joi.string().valid('active', 'inactive', 'completed', 'archived')
});

router.get('/', authenticate, apprenticeController.list);
router.get('/:id', authenticate, apprenticeController.getById);
router.get('/:id/progress', authenticate, apprenticeController.getProgress);
router.post('/', authenticate, isAdmin, validate(createApprenticeSchema), apprenticeController.create);
router.put('/:id', authenticate, isAdmin, validate(updateApprenticeSchema), apprenticeController.update);

module.exports = router;
