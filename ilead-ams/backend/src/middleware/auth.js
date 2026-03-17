const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.JWT_SECRET || 'dev-secret-key-change-in-production';

const authenticate = (req, res, next) => {
  const authHeader = req.get('Authorization');

  if (!authHeader) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Missing authentication token',
      timestamp: new Date().toISOString()
    });
  }

  const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : authHeader;

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    if (err.name === 'TokenExpiredError') {
      return res.status(401).json({
        error: 'Unauthorized',
        message: 'Token expired',
        code: 'TOKEN_EXPIRED',
        timestamp: new Date().toISOString()
      });
    }

    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Invalid token',
      timestamp: new Date().toISOString()
    });
  }
};

const optionalAuth = (req, res, next) => {
  const authHeader = req.get('Authorization');

  if (authHeader) {
    const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : authHeader;
    try {
      const decoded = jwt.verify(token, JWT_SECRET);
      req.user = decoded;
    } catch (err) {
      // Silently ignore auth errors for optional auth
    }
  }

  next();
};

module.exports = {
  authenticate,
  optionalAuth
};
