import express from 'express';
import { authenticate, authorize } from '../middleware/auth.js';

const router = express.Router();

// All bulk operations require admin role
router.use(authenticate);
router.use(authorize(['administrator', 'super_admin']));

/**
 * POST /api/bulk/approve-hours - Bulk approve hour logs
 */
router.post('/approve-hours', async (req, res) => {
  try {
    const { hourLogIds } = req.body;

    if (!hourLogIds || hourLogIds.length === 0) {
      return res.status(400).json({ error: 'No hour logs selected' });
    }

    // Import HourLog model
    const HourLog = (await import('../models/HourLog.js')).default;

    const result = await HourLog.updateMany(
      { _id: { $in: hourLogIds } },
      { status: 'approved', approvedBy: req.user.id, approvedAt: new Date() }
    );

    res.json({
      message: `${result.modifiedCount} hour logs approved`,
      modifiedCount: result.modifiedCount
    });
  } catch (error) {
    console.error('Error bulk approving hours:', error);
    res.status(500).json({ error: 'Bulk approval failed' });
  }
});

/**
 * POST /api/bulk/reject-hours - Bulk reject hour logs
 */
router.post('/reject-hours', async (req, res) => {
  try {
    const { hourLogIds, reason } = req.body;

    if (!hourLogIds || hourLogIds.length === 0) {
      return res.status(400).json({ error: 'No hour logs selected' });
    }

    const HourLog = (await import('../models/HourLog.js')).default;

    const result = await HourLog.updateMany(
      { _id: { $in: hourLogIds } },
      { 
        status: 'rejected',
        rejectionReason: reason || 'Rejected by administrator',
        rejectedBy: req.user.id,
        rejectedAt: new Date()
      }
    );

    res.json({
      message: `${result.modifiedCount} hour logs rejected`,
      modifiedCount: result.modifiedCount
    });
  } catch (error) {
    console.error('Error bulk rejecting hours:', error);
    res.status(500).json({ error: 'Bulk rejection failed' });
  }
});

/**
 * POST /api/bulk/import-apprentices - Bulk import apprentices
 */
router.post('/import-apprentices', async (req, res) => {
  try {
    const { data } = req.body;

    if (!data || data.length === 0) {
      return res.status(400).json({ error: 'No data provided' });
    }

    const Apprentice = (await import('../models/Apprentice.js')).default;
    const results = [];

    for (let i = 0; i < data.length; i++) {
      try {
        const row = data[i];
        
        const apprentice = new Apprentice({
          name: row.name,
          email: row.email,
          program: row.program,
          status: row.status || 'active',
          ojtHours: parseInt(row.ojtHours) || 0
        });

        await apprentice.save();
        results.push({ row: i + 2, success: true, id: apprentice._id });
      } catch (error) {
        results.push({ row: i + 2, success: false, error: error.message });
      }
    }

    const successful = results.filter(r => r.success).length;
    const failed = results.filter(r => !r.success).length;

    res.json({
      message: `Import completed: ${successful} successful, ${failed} failed`,
      results,
      summary: { successful, failed, total: results.length }
    });
  } catch (error) {
    console.error('Error bulk importing apprentices:', error);
    res.status(500).json({ error: 'Import failed' });
  }
});

/**
 * POST /api/bulk/update-status - Bulk update apprentice status
 */
router.post('/update-status', async (req, res) => {
  try {
    const { apprenticeIds, status } = req.body;

    if (!apprenticeIds || apprenticeIds.length === 0) {
      return res.status(400).json({ error: 'No apprentices selected' });
    }

    const validStatuses = ['active', 'completed', 'archived'];
    if (!validStatuses.includes(status)) {
      return res.status(400).json({ error: 'Invalid status' });
    }

    const Apprentice = (await import('../models/Apprentice.js')).default;

    const result = await Apprentice.updateMany(
      { _id: { $in: apprenticeIds } },
      { status }
    );

    res.json({
      message: `${result.modifiedCount} apprentices updated`,
      modifiedCount: result.modifiedCount
    });
  } catch (error) {
    console.error('Error updating statuses:', error);
    res.status(500).json({ error: 'Status update failed' });
  }
});

export default router;
