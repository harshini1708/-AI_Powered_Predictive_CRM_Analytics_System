import client from './client';

/**
 * Upload document
 */
export const uploadDocument = async (file, metadata = {}) => {
  const formData = new FormData();
  formData.append('file', file);

  Object.keys(metadata).forEach(key => {
    formData.append(key, metadata[key]);
  });

  const response = await client.post('/documents', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
  return response.data;
};

/**
 * List documents
 */
export const listDocuments = async (filters = {}) => {
  const params = new URLSearchParams();
  if (filters.apprenticeId) params.append('apprenticeId', filters.apprenticeId);
  if (filters.type) params.append('type', filters.type);
  if (filters.limit) params.append('limit', filters.limit);

  const response = await client.get(`/documents${params ? `?${params}` : ''}`);
  return response.data;
};

/**
 * Get document details
 */
export const getDocument = async (id) => {
  const response = await client.get(`/documents/${id}`);
  return response.data;
};

/**
 * Download document
 */
export const downloadDocument = async (id) => {
  const response = await client.get(`/documents/${id}/download`, {
    responseType: 'blob'
  });
  return response.data;
};

/**
 * Delete document
 */
export const deleteDocument = async (id) => {
  const response = await client.delete(`/documents/${id}`);
  return response.data;
};
