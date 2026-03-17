const ROLES = {
  APPRENTICE: 'apprentice',
  SUPERVISOR: 'supervisor',
  JOURNEYWORKER: 'journeyworker',
  ADMINISTRATOR: 'administrator',
  SUPER_ADMIN: 'super_admin'
};

const roleCheck = (...allowedRoles) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        error: 'Unauthorized',
        message: 'Authentication required',
        timestamp: new Date().toISOString()
      });
    }

    if (!allowedRoles.includes(req.user.role)) {
      return res.status(403).json({
        error: 'Forbidden',
        message: `This action requires one of the following roles: ${allowedRoles.join(', ')}`,
        timestamp: new Date().toISOString()
      });
    }

    next();
  };
};

const isSuperAdmin = (req, res, next) => {
  if (!req.user) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Authentication required',
      timestamp: new Date().toISOString()
    });
  }

  // Check if user is super admin (requires both role AND pinVerified)
  if (req.user.role !== ROLES.SUPER_ADMIN || !req.user.pinVerified) {
    return res.status(403).json({
      error: 'Forbidden',
      message: 'Super Admin access required',
      timestamp: new Date().toISOString()
    });
  }

  next();
};

const isAdmin = (req, res, next) => {
  if (!req.user) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Authentication required',
      timestamp: new Date().toISOString()
    });
  }

  if (![ROLES.ADMINISTRATOR, ROLES.SUPER_ADMIN].includes(req.user.role)) {
    return res.status(403).json({
      error: 'Forbidden',
      message: 'Administrator access required',
      timestamp: new Date().toISOString()
    });
  }

  next();
};

const isReviewerRole = (req, res, next) => {
  if (!req.user) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Authentication required',
      timestamp: new Date().toISOString()
    });
  }

  const reviewerRoles = [ROLES.SUPERVISOR, ROLES.JOURNEYWORKER, ROLES.ADMINISTRATOR, ROLES.SUPER_ADMIN];
  if (!reviewerRoles.includes(req.user.role)) {
    return res.status(403).json({
      error: 'Forbidden',
      message: 'Supervisor, Journeyworker, or Administrator access required',
      timestamp: new Date().toISOString()
    });
  }

  next();
};

module.exports = {
  ROLES,
  roleCheck,
  isSuperAdmin,
  isAdmin,
  isReviewerRole
};
