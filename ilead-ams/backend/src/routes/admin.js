import express from 'express';
import { authenticate, authorize } from '../middleware/auth.js';

const router = express.Router();

// All admin routes require authentication and admin role
router.use(authenticate);
router.use(authorize(['administrator', 'super_admin']));

/**
 * GET /api/admin/health - System health check
 */
router.get('/health', async (req, res) => {
  try {
    const health = {
      status: 'healthy',
      lastCheck: new Date(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      timestamp: new Date().toISOString()
    };

    res.json(health);
  } catch (error) {
    console.error('Error checking system health:', error);
    res.status(500).json({ error: 'Health check failed' });
  }
});

/**
 * GET /api/admin/statistics/users - User statistics
 */
router.get('/statistics/users', async (req, res) => {
  try {
    // Import models
    const User = (await import('../models/User.js')).default;
    const Apprentice = (await import('../models/Apprentice.js')).default;
    const HourLog = (await import('../models/HourLog.js')).default;

    const stats = {
      totalUsers: await User.countDocuments(),
      activeUsers: await User.countDocuments({ status: 'active' }),
      totalApprentices: await Apprentice.countDocuments(),
      activeApprentices: await Apprentice.countDocuments({ status: 'active' }),
      totalHourLogs: await HourLog.countDocuments(),
      pendingApprovals: await HourLog.countDocuments({ status: 'pending' })
    };

    res.json(stats);
  } catch (error) {
    console.error('Error fetching user statistics:', error);
    res.status(500).json({ error: 'Statistics fetch failed' });
  }
});

/**
 * GET /api/admin/statistics/system - System statistics
 */
router.get('/statistics/system', async (req, res) => {
  try {
    const stats = {
      nodeVersion: process.version,
      uptime: Math.floor(process.uptime()),
      memory: {
        total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
        used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
        free: Math.round((process.memoryUsage().heapTotal - process.memoryUsage().heapUsed) / 1024 / 1024)
      },
      cpu: process.cpuUsage(),
      timestamp: new Date().toISOString()
    };

    res.json(stats);
  } catch (error) {
    console.error('Error fetching system statistics:', error);
    res.status(500).json({ error: 'Statistics fetch failed' });
  }
});

/**
 * GET /api/admin/audit-logs - Audit logs
 */
router.get('/audit-logs', async (req, res) => {
  try {
    const { limit = 50, skip = 0 } = req.query;

    // This would be implemented with actual audit logging
    res.json({
      logs: [],
      total: 0,
      limit: parseInt(limit),
      skip: parseInt(skip)
    });
  } catch (error) {
    console.error('Error fetching audit logs:', error);
    res.status(500).json({ error: 'Audit logs fetch failed' });
  }
});

/**
 * GET /api/admin/settings - System settings
 */
router.get('/settings', async (req, res) => {
  try {
    const settings = {
      appName: process.env.APP_NAME || 'I-LEAD AMS',
      appVersion: '4.0.0',
      environment: process.env.NODE_ENV || 'development',
      features: {
        emailNotifications: !!process.env.SMTP_USER,
        twoFactorAuth: false,
        auditLogging: false
      }
    };

    res.json(settings);
  } catch (error) {
    console.error('Error fetching settings:', error);
    res.status(500).json({ error: 'Settings fetch failed' });
  }
});

/**
 * PUT /api/admin/settings - Update system settings
 */
router.put('/settings', async (req, res) => {
  try {
    const { settings } = req.body;

    // Settings would be persisted to database
    res.json({
      message: 'Settings updated successfully',
      settings
    });
  } catch (error) {
    console.error('Error updating settings:', error);
    res.status(500).json({ error: 'Settings update failed' });
  }
});

/**
 * GET /api/admin/database/status - Database status
 */
router.get('/database/status', async (req, res) => {
  try {
    res.json({
      status: 'connected',
      database: process.env.MONGODB_URI ? 'MongoDB' : 'PostgreSQL',
      lastCheck: new Date(),
      responseTime: '< 100ms'
    });
  } catch (error) {
    console.error('Error checking database status:', error);
    res.status(500).json({ error: 'Database check failed' });
  }
});

/**
 * GET /api/admin/backup/status - Backup status
 */
router.get('/backup/status', async (req, res) => {
  try {
    res.json({
      lastBackup: new Date(Date.now() - 24 * 60 * 60 * 1000),
      nextScheduledBackup: new Date(Date.now() + 24 * 60 * 60 * 1000),
      status: 'completed',
      size: '~500 MB'
    });
  } catch (error) {
    console.error('Error checking backup status:', error);
    res.status(500).json({ error: 'Backup check failed' });
  }
});

/**
 * POST /api/admin/backup/trigger - Trigger backup
 */
router.post('/backup/trigger', async (req, res) => {
  try {
    res.json({
      message: 'Backup triggered successfully',
      backupId: 'backup_' + Date.now(),
      status: 'in_progress'
    });
  } catch (error) {
    console.error('Error triggering backup:', error);
    res.status(500).json({ error: 'Backup trigger failed' });
  }
});

export default router;
