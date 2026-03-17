const express = require('express');
const router = express.Router();
const Joi = require('joi');
const { validate } = require('../middleware/validation');
const { authenticate } = require('../middleware/auth');
const authController = require('../controllers/authController');

// Validation Schemas
const registerSchema = Joi.object({
  email: Joi.string().email().required().lowercase().trim(),
  password: Joi.string().min(8).required(),
  firstName: Joi.string().required().trim(),
  lastName: Joi.string().required().trim(),
  phone: Joi.string().allow('').trim(),
  role: Joi.string().valid('apprentice', 'supervisor', 'journeyworker', 'administrator').required(),
  programCode: Joi.string().allow('').trim()
});

const loginSchema = Joi.object({
  email: Joi.string().email().required().lowercase().trim(),
  password: Joi.string().required()
});

const verifyPinSchema = Joi.object({
  pin: Joi.string().length(6).required()
});

const resetPasswordSchema = Joi.object({
  email: Joi.string().email().required().lowercase().trim()
});

const confirmPasswordSchema = Joi.object({
  token: Joi.string().required(),
  password: Joi.string().min(8).required()
});

// Routes
router.post('/register', validate(registerSchema), authController.register);
router.post('/login', validate(loginSchema), authController.login);
router.post('/logout', authenticate, authController.logout);
router.post('/verify-pin', authenticate, validate(verifyPinSchema), authController.verifyPin);
router.post('/refresh-token', authController.refreshToken);
router.post('/forgot-password', validate(resetPasswordSchema), authController.forgotPassword);
router.post('/reset-password', validate(confirmPasswordSchema), authController.resetPassword);
router.get('/me', authenticate, authController.getCurrentUser);

module.exports = router;
