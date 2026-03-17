const express = require('express');
const router = express.Router();
const Joi = require('joi');
const { authenticate } = require('../middleware/auth');
const { isAdmin, isSuperAdmin } = require('../middleware/roleCheck');
const { validate } = require('../middleware/validation');
const userController = require('../controllers/userController');

// Validation schemas
const createUserSchema = Joi.object({
  email: Joi.string().email().required().lowercase().trim(),
  password: Joi.string().min(8).required(),
  firstName: Joi.string().required().trim(),
  lastName: Joi.string().required().trim(),
  phone: Joi.string().allow('').trim(),
  role: Joi.string().valid('apprentice', 'supervisor', 'journeyworker', 'administrator').required()
});

const updateUserSchema = Joi.object({
  firstName: Joi.string().trim(),
  lastName: Joi.string().trim(),
  phone: Joi.string().allow('').trim(),
  role: Joi.string().valid('apprentice', 'supervisor', 'journeyworker', 'administrator', 'super_admin')
});

// Routes
router.get('/', authenticate, isAdmin, userController.list);
router.get('/pending-approvals', authenticate, isAdmin, userController.getPendingApprovals);
router.get('/:id', authenticate, userController.getById);
router.post('/', authenticate, isSuperAdmin, validate(createUserSchema), userController.create);
router.put('/:id', authenticate, validate(updateUserSchema), userController.update);
router.delete('/:id', authenticate, isSuperAdmin, userController.delete);
router.put('/:id/approve', authenticate, isAdmin, userController.approve);
router.put('/:id/deactivate', authenticate, isAdmin, userController.deactivate);

module.exports = router;
