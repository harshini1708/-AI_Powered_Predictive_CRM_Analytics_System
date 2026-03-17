import express from 'express';
import { authenticate } from '../middleware/auth.js';
import reportService from '../services/reportService.js';

const router = express.Router();

router.use(authenticate);

/**
 * GET /api/reports - Get user's reports
 */
router.get('/', async (req, res) => {
  try {
    const { status, type, limit = 20, skip = 0 } = req.query;

    const result = await reportService.getUserReports(req.user.id, {
      status,
      type,
      limit: parseInt(limit),
      skip: parseInt(skip)
    });

    res.json(result);
  } catch (error) {
    console.error('Error fetching reports:', error);
    res.status(500).json({ error: 'Failed to fetch reports' });
  }
});

/**
 * GET /api/reports/:id - Get single report
 */
router.get('/:id', async (req, res) => {
  try {
    const report = await reportService.getReport(req.params.id, req.user.id);
    res.json(report);
  } catch (error) {
    if (error.message === 'Unauthorized') {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    if (error.message === 'Report not found') {
      return res.status(404).json({ error: 'Report not found' });
    }
    console.error('Error fetching report:', error);
    res.status(500).json({ error: 'Failed to fetch report' });
  }
});

/**
 * GET /api/reports/:id/data - Generate report data
 */
router.get('/:id/data', async (req, res) => {
  try {
    const report = await reportService.getReport(req.params.id, req.user.id);
    const data = await reportService.generateReportData(report);
    res.json(data);
  } catch (error) {
    console.error('Error generating report data:', error);
    res.status(500).json({ error: 'Failed to generate report' });
  }
});

/**
 * POST /api/reports - Create new report
 */
router.post('/', async (req, res) => {
  try {
    const reportData = req.body;

    const report = await reportService.createReport(req.user.id, reportData);

    res.status(201).json(report);
  } catch (error) {
    console.error('Error creating report:', error);
    res.status(400).json({ error: 'Failed to create report' });
  }
});

/**
 * PUT /api/reports/:id - Update report
 */
router.put('/:id', async (req, res) => {
  try {
    const report = await reportService.updateReport(
      req.params.id,
      req.user.id,
      req.body
    );

    res.json(report);
  } catch (error) {
    if (error.message === 'Unauthorized') {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    console.error('Error updating report:', error);
    res.status(400).json({ error: 'Failed to update report' });
  }
});

/**
 * DELETE /api/reports/:id - Delete report
 */
router.delete('/:id', async (req, res) => {
  try {
    await reportService.deleteReport(req.params.id, req.user.id);
    res.json({ message: 'Report deleted' });
  } catch (error) {
    if (error.message === 'Unauthorized') {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    console.error('Error deleting report:', error);
    res.status(500).json({ error: 'Failed to delete report' });
  }
});

/**
 * POST /api/reports/:id/share - Share report with user
 */
router.post('/:id/share', async (req, res) => {
  try {
    const { targetUserId, permission } = req.body;

    const report = await reportService.shareReport(
      req.params.id,
      req.user.id,
      targetUserId,
      permission
    );

    res.json(report);
  } catch (error) {
    if (error.message === 'Unauthorized') {
      return res.status(403).json({ error: 'Unauthorized' });
    }
    console.error('Error sharing report:', error);
    res.status(400).json({ error: 'Failed to share report' });
  }
});

/**
 * GET /api/reports/templates - Get available report templates
 */
router.get('/templates', (req, res) => {
  const templates = [
    {
      id: 'hours_summary',
      name: 'Hours Summary',
      description: 'Summary of logged hours by apprentice',
      icon: 'clock'
    },
    {
      id: 'apprentice_progress',
      name: 'Apprentice Progress',
      description: 'Progress towards completion goals',
      icon: 'chart-bar'
    },
    {
      id: 'supervisor_workload',
      name: 'Supervisor Workload',
      description: 'Pending and approved hours by supervisor',
      icon: 'users'
    },
    {
      id: 'program_statistics',
      name: 'Program Statistics',
      description: 'Statistics by training program',
      icon: 'bar-chart'
    },
    {
      id: 'completion_forecast',
      name: 'Completion Forecast',
      description: 'Estimated completion dates',
      icon: 'calendar'
    }
  ];

  res.json(templates);
});

export default router;
