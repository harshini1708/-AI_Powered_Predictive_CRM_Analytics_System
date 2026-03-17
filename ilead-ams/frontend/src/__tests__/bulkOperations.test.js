import {
  parseCSVFile,
  validateCSVData,
  generateImportReport,
  bulkApproveHours,
  bulkRejectHours,
  bulkImportApprentices
} from '../utils/bulkOperations';

describe('Bulk Operations Utility', () => {
  describe('parseCSVFile', () => {
    it('should parse CSV file correctly', async () => {
      const csvContent = `name,email,program
John Doe,john@example.com,HVAC
Jane Smith,jane@example.com,Plumbing`;

      const file = { content: csvContent };
      
      // Mock FileReader
      global.FileReader = class {
        readAsText(f) {
          setTimeout(() => {
            this.onload({ target: { result: csvContent } });
          }, 0);
        }
      };

      const result = await parseCSVFile(file);
      expect(result).toHaveLength(2);
    });
  });

  describe('validateCSVData', () => {
    it('should validate CSV data with required fields', () => {
      const data = [
        { name: 'John', email: 'john@example.com', program: 'HVAC' }
      ];

      const result = validateCSVData(data, ['name', 'email', 'program']);
      expect(result.valid).toBe(true);
    });

    it('should report missing required fields', () => {
      const data = [
        { name: 'John', program: 'HVAC' }
      ];

      const result = validateCSVData(data, ['name', 'email', 'program']);
      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });

    it('should reject empty data', () => {
      const result = validateCSVData([], ['name', 'email']);
      expect(result.valid).toBe(false);
    });
  });

  describe('generateImportReport', () => {
    it('should generate report with summary', () => {
      const results = [
        { row: 2, success: true, id: 'id1' },
        { row: 3, success: true, id: 'id2' },
        { row: 4, success: false, error: 'Email exists' }
      ];

      const report = generateImportReport(results);
      expect(report.total).toBe(3);
      expect(report.successful).toBe(2);
      expect(report.failed).toBe(1);
    });
  });

  describe('Bulk API Operations', () => {
    const mockApiClient = { post: jest.fn() };

    beforeEach(() => jest.clearAllMocks());

    it('should bulk approve hours', async () => {
      mockApiClient.post.mockResolvedValue({
        data: { modifiedCount: 5 }
      });

      const result = await bulkApproveHours(['id1', 'id2'], mockApiClient);
      expect(result.modifiedCount).toBe(5);
    });

    it('should bulk reject hours', async () => {
      mockApiClient.post.mockResolvedValue({
        data: { modifiedCount: 3 }
      });

      const result = await bulkRejectHours(['id1', 'id2'], 'Missing docs', mockApiClient);
      expect(result.modifiedCount).toBe(3);
    });
  });
});
