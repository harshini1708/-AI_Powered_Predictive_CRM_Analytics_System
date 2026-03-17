const errorHandler = (err, req, res, next) => {
  console.error('Error:', {
    message: err.message,
    stack: err.stack,
    path: req.path,
    method: req.method,
    timestamp: new Date().toISOString()
  });

  // Joi Validation Error
  if (err.isJoi) {
    return res.status(400).json({
      error: 'Validation Error',
      details: err.details.map(d => ({
        field: d.path.join('.'),
        message: d.message,
        type: d.type
      })),
      timestamp: new Date().toISOString()
    });
  }

  // Database Errors
  if (err.code === '23505') { // Unique violation
    return res.status(409).json({
      error: 'Conflict',
      message: 'Record already exists',
      timestamp: new Date().toISOString()
    });
  }

  if (err.code === '23503') { // Foreign key violation
    return res.status(400).json({
      error: 'Bad Request',
      message: 'Invalid reference to related record',
      timestamp: new Date().toISOString()
    });
  }

  // JWT Errors
  if (err.name === 'JsonWebTokenError') {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Invalid token',
      timestamp: new Date().toISOString()
    });
  }

  if (err.name === 'TokenExpiredError') {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Token expired',
      timestamp: new Date().toISOString()
    });
  }

  // Custom Application Errors
  if (err.statusCode) {
    return res.status(err.statusCode).json({
      error: err.error || 'Error',
      message: err.message,
      timestamp: new Date().toISOString()
    });
  }

  // Default Error
  res.status(500).json({
    error: 'Internal Server Error',
    message: process.env.NODE_ENV === 'production' ? 'An unexpected error occurred' : err.message,
    timestamp: new Date().toISOString()
  });
};

const asyncHandler = fn => (req, res, next) => {
  Promise.resolve(fn(req, res, next)).catch(next);
};

module.exports = {
  errorHandler,
  asyncHandler
};
