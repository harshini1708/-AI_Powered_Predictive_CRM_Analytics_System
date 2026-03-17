import reportService from '../services/reportService.js';
import Report from '../models/Report.js';

jest.mock('../models/Report.js');

describe('Report Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Create Report', () => {
    it('should create a new report', async () => {
      const mockReport = {
        _id: 'report-id',
        name: 'Test Report',
        createdBy: 'user-id',
        owner: 'user-id',
        save: jest.fn().mockResolvedValue(true)
      };

      Report.mockImplementation(() => mockReport);

      const result = await reportService.createReport('user-id', {
        name: 'Test Report',
        type: 'hours_summary'
      });

      expect(mockReport.save).toHaveBeenCalled();
      expect(result).toBeDefined();
    });

    it('should set createdBy and owner', async () => {
      const mockReport = {
        save: jest.fn().mockResolvedValue(true)
      };

      Report.mockImplementation((data) => {
        expect(data.createdBy).toBe('user-id');
        expect(data.owner).toBe('user-id');
        return mockReport;
      });

      await reportService.createReport('user-id', { name: 'Test' });
    });
  });

  describe('Get User Reports', () => {
    it('should retrieve user reports with pagination', async () => {
      const mockReports = [
        { _id: '1', name: 'Report 1', owner: 'user-id' },
        { _id: '2', name: 'Report 2', owner: 'user-id' }
      ];

      Report.find = jest.fn().mockReturnValue({
        sort: jest.fn().mockReturnValue({
          limit: jest.fn().mockReturnValue({
            skip: jest.fn().mockReturnValue({
              populate: jest.fn().mockReturnValue({
                populate: jest.fn().mockReturnValue({
                  lean: jest.fn().mockResolvedValue(mockReports)
                })
              })
            })
          })
        })
      });

      Report.countDocuments = jest.fn().mockResolvedValue(2);

      const result = await reportService.getUserReports('user-id');

      expect(result.reports).toHaveLength(2);
      expect(result.total).toBe(2);
    });

    it('should filter by status', async () => {
      Report.find = jest.fn().mockReturnValue({
        sort: jest.fn().mockReturnValue({
          limit: jest.fn().mockReturnValue({
            skip: jest.fn().mockReturnValue({
              populate: jest.fn().mockReturnValue({
                populate: jest.fn().mockReturnValue({
                  lean: jest.fn().mockResolvedValue([])
                })
              })
            })
          })
        })
      });

      Report.countDocuments = jest.fn().mockResolvedValue(0);

      await reportService.getUserReports('user-id', { status: 'active' });

      expect(Report.find).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'active'
        })
      );
    });
  });

  describe('Generate Report Data', () => {
    it('should generate hours summary report', async () => {
      const report = {
        type: 'hours_summary',
        filters: { dateRange: {} },
        groupBy: { field: 'none' },
        sortBy: {}
      };

      const result = await reportService.generateReportData(report);

      expect(result).toBeDefined();
      expect(result.rowCount).toBeDefined();
      expect(result.generatedAt).toBeDefined();
    });

    it('should apply grouping to data', () => {
      const data = [
        { program: 'HVAC', hours: 100 },
        { program: 'HVAC', hours: 50 },
        { program: 'Plumbing', hours: 75 }
      ];

      const grouped = reportService.groupData(data, 'program');

      expect(grouped).toHaveLength(2);
      expect(grouped[0].program).toBe('HVAC');
      expect(grouped[0].items).toHaveLength(2);
    });

    it('should apply sorting to data', () => {
      const data = [
        { name: 'John', hours: 100 },
        { name: 'Alice', hours: 50 },
        { name: 'Bob', hours: 75 }
      ];

      const sorted = reportService.sortData(data, 'hours', 'asc');

      expect(sorted[0].name).toBe('Alice');
      expect(sorted[1].name).toBe('Bob');
      expect(sorted[2].name).toBe('John');
    });
  });

  describe('Update Report', () => {
    it('should update report for owner', async () => {
      const mockReport = {
        owner: { toString: () => 'user-id' },
        createdBy: { toString: () => 'other-user' },
        save: jest.fn().mockResolvedValue(true)
      };

      Report.findById = jest.fn().mockResolvedValue(mockReport);

      const result = await reportService.updateReport(
        'report-id',
        'user-id',
        { name: 'Updated Name' }
      );

      expect(mockReport.save).toHaveBeenCalled();
      expect(mockReport.name).toBe('Updated Name');
    });

    it('should deny unauthorized update', async () => {
      const mockReport = {
        owner: { toString: () => 'other-user' },
        createdBy: { toString: () => 'other-user' }
      };

      Report.findById = jest.fn().mockResolvedValue(mockReport);

      await expect(
        reportService.updateReport('report-id', 'user-id', {})
      ).rejects.toThrow('Unauthorized');
    });
  });

  describe('Delete Report', () => {
    it('should delete report for owner', async () => {
      const mockReport = {
        owner: { toString: () => 'user-id' }
      };

      Report.findById = jest.fn().mockResolvedValue(mockReport);
      Report.deleteOne = jest.fn().mockResolvedValue({ deletedCount: 1 });

      const result = await reportService.deleteReport('report-id', 'user-id');

      expect(result).toBe(true);
      expect(Report.deleteOne).toHaveBeenCalled();
    });

    it('should deny unauthorized delete', async () => {
      const mockReport = {
        owner: { toString: () => 'other-user' }
      };

      Report.findById = jest.fn().mockResolvedValue(mockReport);

      await expect(
        reportService.deleteReport('report-id', 'user-id')
      ).rejects.toThrow('Unauthorized');
    });
  });

  describe('Share Report', () => {
    it('should share report with new user', async () => {
      const mockReport = {
        owner: { toString: () => 'user-id' },
        sharedWith: [],
        save: jest.fn().mockResolvedValue(true)
      };

      Report.findById = jest.fn().mockResolvedValue(mockReport);

      await reportService.shareReport('report-id', 'user-id', 'target-user', 'view');

      expect(mockReport.sharedWith).toHaveLength(1);
      expect(mockReport.sharedWith[0].userId).toBe('target-user');
      expect(mockReport.save).toHaveBeenCalled();
    });

    it('should update permission for existing shared user', async () => {
      const mockReport = {
        owner: { toString: () => 'user-id' },
        sharedWith: [
          { userId: { toString: () => 'target-user' }, permission: 'view' }
        ],
        save: jest.fn().mockResolvedValue(true)
      };

      Report.findById = jest.fn().mockResolvedValue(mockReport);

      await reportService.shareReport('report-id', 'user-id', 'target-user', 'edit');

      expect(mockReport.sharedWith[0].permission).toBe('edit');
      expect(mockReport.save).toHaveBeenCalled();
    });
  });

  describe('Get Scheduled Reports', () => {
    it('should retrieve reports due for execution', async () => {
      const mockReports = [
        { _id: '1', name: 'Report 1', schedule: { enabled: true } }
      ];

      Report.find = jest.fn().mockResolvedValue(mockReports);

      const result = await reportService.getScheduledReportsForExecution();

      expect(result).toHaveLength(1);
      expect(Report.find).toHaveBeenCalledWith(
        expect.objectContaining({
          'schedule.enabled': true,
          'schedule.nextRun': expect.any(Object)
        })
      );
    });
  });
});
