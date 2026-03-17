/**
 * Bulk Operations Utility
 * Handles bulk actions like approvals, rejections, CSV imports
 */

/**
 * Bulk approve hour logs
 */
export const bulkApproveHours = async (hourLogIds, apiClient) => {
  try {
    const response = await apiClient.post('/hour-logs/bulk/approve', {
      hourLogIds
    });
    return response.data;
  } catch (error) {
    throw new Error(`Bulk approval failed: ${error.message}`);
  }
};

/**
 * Bulk reject hour logs
 */
export const bulkRejectHours = async (hourLogIds, reason, apiClient) => {
  try {
    const response = await apiClient.post('/hour-logs/bulk/reject', {
      hourLogIds,
      reason
    });
    return response.data;
  } catch (error) {
    throw new Error(`Bulk rejection failed: ${error.message}`);
  }
};

/**
 * Parse CSV file
 */
export const parseCSVFile = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      try {
        const text = e.target.result;
        const lines = text.split('\n');
        const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
        
        const data = [];
        for (let i = 1; i < lines.length; i++) {
          if (lines[i].trim() === '') continue;
          
          const values = lines[i].split(',').map(v => v.trim());
          const row = {};
          
          headers.forEach((header, index) => {
            row[header] = values[index] || '';
          });
          
          data.push(row);
        }

        resolve(data);
      } catch (error) {
        reject(new Error(`CSV parsing failed: ${error.message}`));
      }
    };

    reader.onerror = () => {
      reject(new Error('File reading failed'));
    };

    reader.readAsText(file);
  });
};

/**
 * Bulk import apprentices
 */
export const bulkImportApprentices = async (csvData, apiClient) => {
  try {
    const response = await apiClient.post('/apprentices/bulk/import', {
      data: csvData
    });
    return response.data;
  } catch (error) {
    throw new Error(`Bulk import failed: ${error.message}`);
  }
};

/**
 * Bulk import users
 */
export const bulkImportUsers = async (csvData, apiClient) => {
  try {
    const response = await apiClient.post('/users/bulk/import', {
      data: csvData
    });
    return response.data;
  } catch (error) {
    throw new Error(`Bulk import failed: ${error.message}`);
  }
};

/**
 * Bulk update apprentice status
 */
export const bulkUpdateApprenticeStatus = async (apprenticeIds, status, apiClient) => {
  try {
    const response = await apiClient.post('/apprentices/bulk/status', {
      apprenticeIds,
      status
    });
    return response.data;
  } catch (error) {
    throw new Error(`Status update failed: ${error.message}`);
  }
};

/**
 * Bulk assign supervisor
 */
export const bulkAssignSupervisor = async (apprenticeIds, supervisorId, apiClient) => {
  try {
    const response = await apiClient.post('/apprentices/bulk/assign-supervisor', {
      apprenticeIds,
      supervisorId
    });
    return response.data;
  } catch (error) {
    throw new Error(`Supervisor assignment failed: ${error.message}`);
  }
};

/**
 * Validate CSV data before import
 */
export const validateCSVData = (data, requiredFields) => {
  const errors = [];

  if (!data || data.length === 0) {
    return { valid: false, errors: ['CSV file is empty'] };
  }

  // Check required fields
  const firstRow = data[0];
  const missingFields = requiredFields.filter(field => !firstRow.hasOwnProperty(field));
  
  if (missingFields.length > 0) {
    errors.push(`Missing required fields: ${missingFields.join(', ')}`);
  }

  // Validate data rows
  data.forEach((row, index) => {
    requiredFields.forEach(field => {
      if (!row[field] || row[field].trim() === '') {
        errors.push(`Row ${index + 2}: Missing value for ${field}`);
      }
    });
  });

  return {
    valid: errors.length === 0,
    errors
  };
};

/**
 * Generate import report
 */
export const generateImportReport = (results) => {
  const report = {
    total: results.length,
    successful: results.filter(r => r.success).length,
    failed: results.filter(r => !r.success).length,
    errors: results.filter(r => !r.success).map(r => ({
      row: r.row,
      error: r.error
    }))
  };

  return report;
};
