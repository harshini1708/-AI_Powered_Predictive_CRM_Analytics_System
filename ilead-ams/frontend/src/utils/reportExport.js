/**
 * Report Export Utilities
 * Handles exporting report data to CSV, Excel, and PDF
 */

/**
 * Export to CSV
 */
export const exportToCSV = (data, filename = 'report.csv') => {
  if (!data || data.length === 0) {
    alert('No data to export');
    return;
  }

  // Get headers from first row
  const headers = Object.keys(data[0]);
  
  // Create CSV content
  let csv = headers.join(',') + '\n';
  
  data.forEach(row => {
    const values = headers.map(header => {
      const value = row[header];
      // Escape quotes and wrap in quotes if contains comma
      if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
        return `"${value.replace(/"/g, '""')}"`;
      }
      return value || '';
    });
    csv += values.join(',') + '\n';
  });

  // Download
  downloadFile(csv, filename, 'text/csv');
};

/**
 * Export to Excel (requires xlsx library)
 */
export const exportToExcel = (data, filename = 'report.xlsx') => {
  try {
    // Check if xlsx is available
    if (typeof window.XLSX === 'undefined') {
      alert('Excel export requires xlsx library. Install with: npm install xlsx');
      return;
    }

    const worksheet = window.XLSX.utils.json_to_sheet(data);
    const workbook = window.XLSX.utils.book_new();
    window.XLSX.utils.book_append_sheet(workbook, worksheet, 'Report');
    window.XLSX.writeFile(workbook, filename);
  } catch (error) {
    console.error('Error exporting to Excel:', error);
    alert('Error exporting to Excel: ' + error.message);
  }
};

/**
 * Export to PDF (requires jsPDF and autoTable)
 */
export const exportToPDF = (data, reportName = 'Report') => {
  try {
    // Check if jsPDF is available
    if (typeof window.jsPDF === 'undefined') {
      alert('PDF export requires jsPDF library. Install with: npm install jspdf jspdf-autotable');
      return;
    }

    const { jsPDF } = window.jsPDF;
    const doc = new jsPDF();

    // Add title
    doc.setFontSize(16);
    doc.text(reportName, 14, 15);

    // Add date
    doc.setFontSize(10);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 25);

    // Add table
    if (data && data.length > 0) {
      const headers = Object.keys(data[0]);
      const rows = data.map(row => headers.map(h => row[h]));

      doc.autoTable({
        head: [headers],
        body: rows,
        startY: 35,
        margin: 10,
        didDrawPage: function(data) {
          // Footer
          const pageSize = doc.internal.pageSize;
          const pageHeight = pageSize.getHeight();
          doc.setFontSize(9);
          doc.text(
            `Page ${doc.internal.getNumberOfPages()}`,
            pageSize.getWidth() / 2,
            pageHeight - 10,
            { align: 'center' }
          );
        }
      });
    }

    // Download
    doc.save(`${reportName}.pdf`);
  } catch (error) {
    console.error('Error exporting to PDF:', error);
    alert('Error exporting to PDF: ' + error.message);
  }
};

/**
 * Download file utility
 */
function downloadFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Generate report summary
 */
export const generateSummary = (data) => {
  if (!data || data.length === 0) {
    return {};
  }

  const summary = {
    totalRows: data.length,
    columns: Object.keys(data[0])
  };

  // Calculate statistics for numeric columns
  Object.keys(data[0]).forEach(key => {
    const values = data.map(row => {
      const val = row[key];
      return typeof val === 'number' ? val : parseFloat(val);
    }).filter(v => !isNaN(v));

    if (values.length > 0) {
      summary[`${key}_sum`] = values.reduce((a, b) => a + b, 0);
      summary[`${key}_avg`] = Math.round(summary[`${key}_sum`] / values.length * 100) / 100;
      summary[`${key}_max`] = Math.max(...values);
      summary[`${key}_min`] = Math.min(...values);
    }
  });

  return summary;
};

/**
 * Format report data for display
 */
export const formatReportData = (data, format = {}) => {
  if (!data) return [];

  return data.map(row => {
    const formatted = {};
    Object.keys(row).forEach(key => {
      let value = row[key];

      // Apply formatting based on column type
      if (format[key]) {
        const { type } = format[key];
        switch (type) {
          case 'currency':
            value = typeof value === 'number' ? `$${value.toFixed(2)}` : value;
            break;
          case 'percentage':
            value = typeof value === 'number' ? `${value.toFixed(2)}%` : value;
            break;
          case 'date':
            value = value ? new Date(value).toLocaleDateString() : '';
            break;
          case 'number':
            value = typeof value === 'number' ? value.toLocaleString() : value;
            break;
          default:
            break;
        }
      }

      formatted[key] = value;
    });
    return formatted;
  });
};
