const Joi = require('joi');

const validate = (schema) => {
  return (req, res, next) => {
    const { error, value } = schema.validate(req.body, {
      abortEarly: false,
      stripUnknown: true
    });

    if (error) {
      return res.status(400).json({
        error: 'Validation Error',
        details: error.details.map(d => ({
          field: d.path.join('.'),
          message: d.message.replace(/"/g, "'"),
          type: d.type
        })),
        timestamp: new Date().toISOString()
      });
    }

    req.body = value;
    next();
  };
};

// Joi Schemas for Validation

const emailSchema = Joi.string().email().required().lowercase().trim();
const passwordSchema = Joi.string().min(8).required();
const uuidSchema = Joi.string().uuid().required();
const isoDateSchema = Joi.string().isoDate().required();

module.exports = {
  validate,
  emailSchema,
  passwordSchema,
  uuidSchema,
  isoDateSchema
};
