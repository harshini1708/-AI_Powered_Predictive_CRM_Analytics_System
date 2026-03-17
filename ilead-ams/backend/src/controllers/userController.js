const User = require('../models/User');
const { hashPassword } = require('../utils/password');

module.exports = {
  // List all users in organization (admin only)
  list: async (req, res) => {
    try {
      const { role, status, search, limit, offset } = req.query;
      const organizationId = req.user.organizationId;

      const filters = {
        role: role || undefined,
        status: status || undefined,
        search: search || undefined,
        limit: limit || 50,
        offset: offset || 0
      };

      const users = await User.list(organizationId, filters);

      res.json({
        data: users.map(u => ({
          id: u.id,
          email: u.email,
          firstName: u.first_name,
          lastName: u.last_name,
          phone: u.phone,
          role: u.role,
          status: u.status,
          createdAt: u.created_at
        })),
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('List users error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // Get user by ID
  getById: async (req, res) => {
    try {
      const { id } = req.params;
      const user = await User.findById(id);

      if (!user) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'User not found',
          timestamp: new Date().toISOString()
        });
      }

      // Authorization: can only view users in same organization
      if (user.organization_id !== req.user.organizationId) {
        return res.status(403).json({
          error: 'Forbidden',
          message: 'Cannot access user from different organization',
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
          createdAt: user.created_at,
          updatedAt: user.updated_at
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Get user error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // Create new user (super admin only)
  create: async (req, res) => {
    try {
      const { email, password, firstName, lastName, phone, role } = req.body;

      // Prevent super_admin creation via API (only in seed)
      if (role === 'super_admin') {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Cannot create super_admin users via API',
          timestamp: new Date().toISOString()
        });
      }

      // Check if user already exists
      const existing = await User.findByEmail(email);
      if (existing) {
        return res.status(409).json({
          error: 'Conflict',
          message: 'User already exists',
          timestamp: new Date().toISOString()
        });
      }

      const user = await User.create({
        email,
        password,
        firstName,
        lastName,
        phone,
        role,
        organizationId: req.user.organizationId,
        createdBy: req.user.id
      });

      // Auto-approve users created by super admin
      if (req.user.role === 'super_admin') {
        await User.approve(user.id);
      }

      res.status(201).json({
        message: 'User created',
        user: {
          id: user.id,
          email: user.email,
          firstName: user.first_name,
          lastName: user.last_name,
          role: user.role,
          status: user.status
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Create user error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // Update user profile
  update: async (req, res) => {
    try {
      const { id } = req.params;
      const { firstName, lastName, phone, role } = req.body;

      const user = await User.findById(id);
      if (!user) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'User not found',
          timestamp: new Date().toISOString()
        });
      }

      // Authorization: users can only update themselves, admins can update others
      if (req.user.id !== id && req.user.role !== 'administrator' && req.user.role !== 'super_admin') {
        return res.status(403).json({
          error: 'Forbidden',
          message: 'Cannot update other users',
          timestamp: new Date().toISOString()
        });
      }

      // Organization check
      if (user.organization_id !== req.user.organizationId) {
        return res.status(403).json({
          error: 'Forbidden',
          message: 'Cannot access user from different organization',
          timestamp: new Date().toISOString()
        });
      }

      // Prevent role change for self (except super admin)
      if (req.user.id === id && role && role !== user.role && req.user.role !== 'super_admin') {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Cannot change your own role',
          timestamp: new Date().toISOString()
        });
      }

      // Prevent non-super-admin from creating/modifying super_admin
      if (role === 'super_admin' && req.user.role !== 'super_admin') {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Only super admin can create super_admin users',
          timestamp: new Date().toISOString()
        });
      }

      const updates = {
        first_name: firstName !== undefined ? firstName : user.first_name,
        last_name: lastName !== undefined ? lastName : user.last_name,
        phone: phone !== undefined ? phone : user.phone
      };

      if (role && req.user.role === 'super_admin') {
        updates.role = role;
      }

      const updatedUser = await User.update(id, updates);

      res.json({
        message: 'User updated',
        user: {
          id: updatedUser.id,
          email: updatedUser.email,
          firstName: updatedUser.first_name,
          lastName: updatedUser.last_name,
          phone: updatedUser.phone,
          role: updatedUser.role,
          status: updatedUser.status
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Update user error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // Delete user (super admin only)
  delete: async (req, res) => {
    try {
      const { id } = req.params;

      const user = await User.findById(id);
      if (!user) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'User not found',
          timestamp: new Date().toISOString()
        });
      }

      // Prevent deleting super admin
      if (user.role === 'super_admin') {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'Cannot delete super admin users',
          timestamp: new Date().toISOString()
        });
      }

      await User.delete(id);

      res.json({
        message: 'User deleted',
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Delete user error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // Get pending approvals (admin only)
  getPendingApprovals: async (req, res) => {
    try {
      const organizationId = req.user.organizationId;

      const pendingUsers = await User.findPendingApprovals(organizationId);

      res.json({
        data: pendingUsers.map(u => ({
          id: u.id,
          email: u.email,
          firstName: u.first_name,
          lastName: u.last_name,
          phone: u.phone,
          role: u.role,
          createdAt: u.created_at
        })),
        count: pendingUsers.length,
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Get pending approvals error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // Approve user (admin only)
  approve: async (req, res) => {
    try {
      const { id } = req.params;

      const user = await User.findById(id);
      if (!user) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'User not found',
          timestamp: new Date().toISOString()
        });
      }

      if (user.status !== 'pending_approval') {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'User is not pending approval',
          timestamp: new Date().toISOString()
        });
      }

      const approvedUser = await User.approve(id);

      res.json({
        message: 'User approved',
        user: {
          id: approvedUser.id,
          email: approvedUser.email,
          firstName: approvedUser.first_name,
          lastName: approvedUser.last_name,
          role: approvedUser.role,
          status: approvedUser.status
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Approve user error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  // Deactivate user (admin only)
  deactivate: async (req, res) => {
    try {
      const { id } = req.params;

      const user = await User.findById(id);
      if (!user) {
        return res.status(404).json({
          error: 'Not Found',
          message: 'User not found',
          timestamp: new Date().toISOString()
        });
      }

      const deactivatedUser = await User.deactivate(id);

      res.json({
        message: 'User deactivated',
        user: {
          id: deactivatedUser.id,
          email: deactivatedUser.email,
          status: deactivatedUser.status
        },
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Deactivate user error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  }
};
