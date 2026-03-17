const User = require('../models/User');
const Organization = require('../models/Organization');
const { verifyPassword, verifyPin } = require('../utils/password');
const { generateAuthTokens, verifyToken } = require('../utils/jwt');
const db = require('../config/database');

const SUPER_ADMIN_EMAIL = process.env.SUPER_ADMIN_EMAIL || 'hector.torres@i-leadusa.org';
const SUPER_ADMIN_PIN_HASH = process.env.SUPER_ADMIN_PIN_HASH;

module.exports = {
  register: async (req, res) => {
    try {
      const { email, password, firstName, lastName, phone, role, programCode } = req.body;

      // Validate role - don't allow super_admin registration
      if (role === 'super_admin') {
        return res.status(400).json({
          error: 'Invalid Role',
          message: 'Cannot register as super_admin',
          timestamp: new Date().toISOString()
        });
      }

      // Check if user already exists
      const existing = await User.findByEmail(email);
      if (existing) {
        return res.status(409).json({
          error: 'Conflict',
          message: 'Email already registered',
          timestamp: new Date().toISOString()
        });
      }

      // Get or create default organization (for now, use ILEAD-001)
      let org = await db('organizations').where('code', 'ILEAD-001').first();
      if (!org) {
        const [newOrg] = await db('organizations')
          .insert({
            id: require('uuid').v4(),
            name: 'I-LEAD Inc.',
            code: 'ILEAD-001',
            address: 'Reading, PA',
            created_at: new Date(),
            updated_at: new Date()
          })
          .returning('*');
        org = newOrg;
      }

      // Create user
      const user = await User.create({
        email,
        password,
        firstName,
        lastName,
        phone,
        role,
        organizationId: org.id
      });

      const tokens = generateAuthTokens(user);

      res.status(201).json({
        message: 'Account created. Pending administrator approval.',
        user: {
          id: user.id,
          email: user.email,
          firstName: user.first_name,
          lastName: user.last_name,
          role: user.role,
          status: user.status
        },
        ...tokens,
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Register error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  login: async (req, res) => {
    try {
      const { email, password } = req.body;

      // Find user
      const user = await User.findByEmail(email);
      if (!user) {
        return res.status(401).json({
          error: 'Unauthorized',
          message: 'Invalid email or password',
          timestamp: new Date().toISOString()
        });
      }

      // Check password
      const validPassword = await verifyPassword(password, user.password_hash);
      if (!validPassword) {
        return res.status(401).json({
          error: 'Unauthorized',
          message: 'Invalid email or password',
          timestamp: new Date().toISOString()
        });
      }

      // Check account status
      if (user.status === 'pending_approval') {
        return res.status(403).json({
          error: 'Forbidden',
          message: 'Account pending administrator approval',
          code: 'ACCOUNT_PENDING_APPROVAL',
          timestamp: new Date().toISOString()
        });
      }

      if (user.status === 'inactive') {
        return res.status(403).json({
          error: 'Forbidden',
          message: 'Account is inactive',
          code: 'ACCOUNT_INACTIVE',
          timestamp: new Date().toISOString()
        });
      }

      // For super admin, require PIN verification
      if (user.role === 'super_admin' || user.email === SUPER_ADMIN_EMAIL) {
        return res.json({
          message: 'Login successful. PIN verification required.',
          user: {
            id: user.id,
            email: user.email,
            firstName: user.first_name,
            lastName: user.last_name,
            role: user.role
          },
          requiresPin: true,
          pinToken: require('jsonwebtoken').sign(
            { userId: user.id, type: 'pin-verification' },
            process.env.JWT_SECRET || 'dev-secret-key-change-in-production',
            { expiresIn: '5m' }
          ),
          timestamp: new Date().toISOString()
        });
      }

      // Generate tokens for regular users
      const tokens = generateAuthTokens(user);

      res.json({
        message: 'Login successful',
        user: {
          id: user.id,
          email: user.email,
          firstName: user.first_name,
          lastName: user.last_name,
          role: user.role
        },
        ...tokens,
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Login error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  verifyPin: async (req, res) => {
    try {
      const { pin } = req.body;
      const { userId } = req.user; // From pin-verification token or auth

      const user = await User.findById(req.user.id || userId);
      if (!user) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'User not found',
          timestamp: new Date().toISOString()
        });
      }

      // Verify PIN against expected super admin PIN
      if (!SUPER_ADMIN_PIN_HASH) {
        // In development, compare against plain text PIN from env
        const superAdminPin = process.env.SUPER_ADMIN_PIN_PLAIN || '071676';
        if (pin !== superAdminPin) {
          return res.status(401).json({
            error: 'Unauthorized',
            message: 'Invalid PIN',
            timestamp: new Date().toISOString()
          });
        }
      } else {
        const validPin = await verifyPin(pin, SUPER_ADMIN_PIN_HASH);
        if (!validPin) {
          return res.status(401).json({
            error: 'Unauthorized',
            message: 'Invalid PIN',
            timestamp: new Date().toISOString()
          });
        }
      }

      // Update user with pin_verified flag (in memory, normally would persist)
      user.pin_verified = true;

      const tokens = generateAuthTokens(user);

      res.json({
        message: 'PIN verified',
        user: {
          id: user.id,
          email: user.email,
          firstName: user.first_name,
          lastName: user.last_name,
          role: user.role
        },
        ...tokens,
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Verify PIN error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  logout: async (req, res) => {
    // Token invalidation would require a blacklist/logout table in production
    res.json({
      message: 'Logged out successfully',
      timestamp: new Date().toISOString()
    });
  },

  refreshToken: async (req, res) => {
    try {
      const { refreshToken } = req.body;

      if (!refreshToken) {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Refresh token required',
          timestamp: new Date().toISOString()
        });
      }

      const decoded = verifyToken(refreshToken);
      if (!decoded || decoded.type !== 'refresh') {
        return res.status(401).json({
          error: 'Unauthorized',
          message: 'Invalid refresh token',
          timestamp: new Date().toISOString()
        });
      }

      const user = await User.findById(decoded.userId);
      if (!user) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'User not found',
          timestamp: new Date().toISOString()
        });
      }

      const tokens = generateAuthTokens(user);

      res.json({
        message: 'Token refreshed',
        ...tokens,
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Refresh token error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  forgotPassword: async (req, res) => {
    try {
      const { email } = req.body;

      const user = await User.findByEmail(email);
      if (!user) {
        // Don't reveal if email exists for security
        return res.json({
          message: 'If account exists, password reset email has been sent',
          timestamp: new Date().toISOString()
        });
      }

      // Generate reset token
      const resetToken = require('crypto').randomBytes(32).toString('hex');
      const resetTokenHash = require('crypto')
        .createHash('sha256')
        .update(resetToken)
        .digest('hex');

      // Store reset token in database
      await db('password_reset_tokens').insert({
        id: require('uuid').v4(),
        user_id: user.id,
        token: resetTokenHash,
        expires_at: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24 hours
        used: false,
        created_at: new Date(),
        updated_at: new Date()
      });

      // In production, send email with reset link
      console.log(`Password reset token for ${email}: ${resetToken}`);

      res.json({
        message: 'If account exists, password reset email has been sent',
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Forgot password error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  resetPassword: async (req, res) => {
    try {
      const { token, password } = req.body;

      const tokenHash = require('crypto')
        .createHash('sha256')
        .update(token)
        .digest('hex');

      // Find valid reset token
      const resetToken = await db('password_reset_tokens')
        .where('token', tokenHash)
        .where('used', false)
        .where('expires_at', '>', new Date())
        .first();

      if (!resetToken) {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Invalid or expired reset token',
          timestamp: new Date().toISOString()
        });
      }

      // Update password
      await User.updatePassword(resetToken.user_id, password);

      // Mark token as used
      await db('password_reset_tokens')
        .where('id', resetToken.id)
        .update({ used: true });

      res.json({
        message: 'Password reset successful',
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Reset password error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  getCurrentUser: async (req, res) => {
    try {
      const user = await User.findById(req.user.id);
      if (!user) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'User not found',
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        user: {
          id: user.id,
          email: user.email,
          firstName: user.first_name,
          lastName: user.last_name,
          phone: user.phone,
          role: user.role,
          status: user.status,
          organizationId: user.organization_id,
          createdAt: user.created_at
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Get current user error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  }
};
