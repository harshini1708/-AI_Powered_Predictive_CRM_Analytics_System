const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.JWT_SECRET || 'dev-secret-key-change-in-production';
const JWT_EXPIRY = process.env.JWT_EXPIRY || '7d';
const JWT_REFRESH_EXPIRY = process.env.JWT_REFRESH_EXPIRY || '30d';

function generateToken(payload, expiresIn = JWT_EXPIRY) {
  return jwt.sign(payload, JWT_SECRET, { expiresIn });
}

function generateRefreshToken(userId) {
  return jwt.sign(
    { userId, type: 'refresh' },
    JWT_SECRET,
    { expiresIn: JWT_REFRESH_EXPIRY }
  );
}

function generateAccessToken(user) {
  return jwt.sign(
    {
      id: user.id,
      email: user.email,
      role: user.role,
      organizationId: user.organization_id,
      pinVerified: user.pin_verified || false
    },
    JWT_SECRET,
    { expiresIn: JWT_EXPIRY }
  );
}

function generateAuthTokens(user) {
  return {
    accessToken: generateAccessToken(user),
    refreshToken: generateRefreshToken(user.id),
    expiresIn: JWT_EXPIRY
  };
}

function verifyToken(token) {
  try {
    return jwt.verify(token, JWT_SECRET);
  } catch (err) {
    return null;
  }
}

function decodeToken(token) {
  try {
    return jwt.decode(token);
  } catch (err) {
    return null;
  }
}

module.exports = {
  generateToken,
  generateRefreshToken,
  generateAccessToken,
  generateAuthTokens,
  verifyToken,
  decodeToken
};
