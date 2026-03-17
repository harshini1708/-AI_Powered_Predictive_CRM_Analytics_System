const documentService = require('../services/documentService');

module.exports = {
  upload: async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          error: 'Bad Request',
          message: 'No file provided',
          timestamp: new Date().toISOString()
        });
      }

      const { apprenticeId } = req.body;
      const organizationId = req.user.organizationId;

      const document = await documentService.uploadDocument(
        req.file.buffer,
        req.file.originalname,
        req.file.mimetype,
        req.user.id,
        organizationId,
        apprenticeId || null
      );

      res.status(201).json({
        message: 'Document uploaded',
        document,
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Upload document error:', err);
      res.status(400).json({
        error: 'Bad Request',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  list: async (req, res) => {
    try {
      const { apprenticeId, limit, offset } = req.query;
      const organizationId = req.user.organizationId;

      const filters = { apprenticeId, limit, offset };
      const documents = await documentService.listDocuments(organizationId, filters);

      res.json({
        data: documents.map(d => ({
          id: d.id,
          filename: d.filename,
          fileSize: d.file_size,
          mimeType: d.mime_type,
          url: d.storage_url,
          uploadedAt: d.created_at
        })),
        count: documents.length,
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('List documents error:', err);
      res.status(500).json({
        error: 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  },

  delete: async (req, res) => {
    try {
      const { id } = req.params;
      const organizationId = req.user.organizationId;

      await documentService.deleteDocument(id, organizationId);

      res.json({
        message: 'Document deleted',
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Delete document error:', err);
      res.status(err.message.includes('not found') ? 404 : 500).json({
        error: err.message.includes('not found') ? 'Not Found' : 'Internal Server Error',
        message: err.message,
        timestamp: new Date().toISOString()
      });
    }
  }
};
