const { v4: uuidv4 } = require('uuid');

function generateId(prefix = '') {
  const uuid = uuidv4();
  if (!prefix) return uuid;
  return `${prefix}-${uuid.substring(0, 8).toUpperCase()}`;
}

function generateUserId() {
  return generateId('USR');
}

function generateApprenticeId() {
  return generateId('APP');
}

function generateHourLogId() {
  return generateId('LOG');
}

function generateOrganizationId() {
  return generateId('ORG');
}

function generateDocumentId() {
  return generateId('DOC');
}

function generateRatingId() {
  return generateId('RAT');
}

module.exports = {
  generateId,
  generateUserId,
  generateApprenticeId,
  generateHourLogId,
  generateOrganizationId,
  generateDocumentId,
  generateRatingId
};
