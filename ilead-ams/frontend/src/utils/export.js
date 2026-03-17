import Papa from 'papaparse';

/**
 * Export data to CSV file
 */
export const exportToCSV = (data, filename = 'export.csv') => {
  try {
    const csv = Papa.unparse(data);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  } catch (error) {
    console.error('CSV export failed:', error);
    throw error;
  }
};

/**
 * Export hour logs to CSV
 */
export const exportHourLogs = (logs) => {
  const data = logs.map(log => ({
    'Date': new Date(log.date).toLocaleDateString(),
    'Apprentice': log.apprenticeName || log.apprenticeId,
    'OJT Domain': log.ojtDomainName || '—',
    'OJT Hours': log.ojtHours || 0,
    'RTI Hours': log.rtiHours || 0,
    'Qualified Hours': log.qualifiedHours || 0,
    'Status': log.status,
    'Submitted By': log.submittedBy,
    'Submitted At': new Date(log.submittedAt).toLocaleString()
  }));

  exportToCSV(data, `hour-logs-${new Date().toISOString().split('T')[0]}.csv`);
};

/**
 * Export apprentices to CSV
 */
export const exportApprentices = (apprentices) => {
  const data = apprentices.map(app => ({
    'Name': app.name,
    'Email': app.email,
    'Program': app.program,
    'Status': app.status,
    'OJT Hours': app.ojtHours || 0,
    'Qualified Hours': app.qualifiedOjtHours || 0,
    'Supervisor': app.supervisor || '—',
    'Employer': app.employer || '—',
    'Start Date': app.startDate ? new Date(app.startDate).toLocaleDateString() : '—'
  }));

  exportToCSV(data, `apprentices-${new Date().toISOString().split('T')[0]}.csv`);
};

/**
 * Export users to CSV
 */
export const exportUsers = (users) => {
  const data = users.map(user => ({
    'Name': `${user.firstName} ${user.lastName}`,
    'Email': user.email,
    'Role': user.role,
    'Status': user.status,
    'Phone': user.phone || '—',
    'Joined': new Date(user.createdAt).toLocaleDateString()
  }));

  exportToCSV(data, `users-${new Date().toISOString().split('T')[0]}.csv`);
};

/**
 * Export domain progress to CSV
 */
export const exportDomainProgress = (domains) => {
  const data = domains.map(domain => ({
    'Domain': domain.domain,
    'Qualified Hours': domain.qualifiedHours || 0
  }));

  exportToCSV(data, `domain-progress-${new Date().toISOString().split('T')[0]}.csv`);
};

/**
 * Generate PDF from HTML element (uses print functionality)
 */
export const generatePDF = (elementId, filename = 'report.pdf') => {
  const element = document.getElementById(elementId);
  if (!element) {
    console.error(`Element with id ${elementId} not found`);
    return;
  }

  const printWindow = window.open('', '', 'height=400,width=800');
  printWindow.document.write('<html><head><title>' + filename + '</title>');
  printWindow.document.write('<style>');
  printWindow.document.write('body { font-family: Arial, sans-serif; margin: 20px; }');
  printWindow.document.write('table { width: 100%; border-collapse: collapse; margin-top: 20px; }');
  printWindow.document.write('th, td { padding: 10px; text-align: left; border: 1px solid #ddd; }');
  printWindow.document.write('th { background-color: #f8f9fa; font-weight: bold; }');
  printWindow.document.write('h1 { color: #003d82; }');
  printWindow.document.write('h2 { color: #003d82; margin-top: 30px; }');
  printWindow.document.write('.timestamp { color: #6c757d; font-size: 12px; margin-top: 20px; }');
  printWindow.document.write('</style></head><body>');
  printWindow.document.write(element.innerHTML);
  printWindow.document.write('<div class="timestamp">Generated on ' + new Date().toLocaleString() + '</div>');
  printWindow.document.write('</body></html>');
  printWindow.document.close();

  setTimeout(() => {
    printWindow.print();
  }, 250);
};
