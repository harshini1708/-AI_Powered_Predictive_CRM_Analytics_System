const fs = require('fs').promises;
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const db = require('../config/database');

class DocumentService {
  constructor() {
    this.storageType = process.env.STORAGE_TYPE || 'local';
    this.uploadDir = process.env.UPLOAD_DIR || path.join(__dirname, '../../uploads');
    this.maxFileSize = parseInt(process.env.MAX_FILE_SIZE) || 50 * 1024 * 1024; // 50MB default
    this.allowedMimeTypes = [
      'application/pdf',
      'image/jpeg',
      'image/png',
      'image/gif',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/msword',
      'text/plain'
    ];

    this.initializeStorage();
  }

  async initializeStorage() {
    if (this.storageType === 'local') {
      try {
        await fs.mkdir(this.uploadDir, { recursive: true });
        console.log(`📁 Local document storage initialized at ${this.uploadDir}`);
      } catch (err) {
        console.error('Failed to initialize upload directory:', err);
      }
    } else if (this.storageType === 's3') {
      console.log('☁️  Using AWS S3 for document storage');
    }
  }

  validateFile(buffer, mimeType, originalName) {
    // Check file size
    if (buffer.length > this.maxFileSize) {
      throw new Error(`File size exceeds ${this.maxFileSize / 1024 / 1024}MB limit`);
    }

    // Check MIME type
    if (!this.allowedMimeTypes.includes(mimeType)) {
      throw new Error(`File type ${mimeType} not allowed`);
    }

    // Check file extension matches MIME type
    const ext = path.extname(originalName).toLowerCase();
    const validExtensions = {
      'application/pdf': ['.pdf'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/gif': ['.gif'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/msword': ['.doc'],
      'text/plain': ['.txt']
    };

    if (!validExtensions[mimeType] || !validExtensions[mimeType].includes(ext)) {
      throw new Error('File extension does not match file type');
    }

    return true;
  }

  async uploadLocal(buffer, originalName) {
    const fileId = uuidv4();
    const ext = path.extname(originalName);
    const fileName = `${fileId}${ext}`;
    const filePath = path.join(this.uploadDir, fileName);

    try {
      await fs.writeFile(filePath, buffer);
      console.log(`✅ File uploaded locally: ${fileName}`);
      return {
        fileId,
        fileName,
        storagePath: filePath,
        storageUrl: `/uploads/${fileName}`,
        size: buffer.length
      };
    } catch (err) {
      console.error('Local file upload failed:', err);
      throw new Error('Failed to upload file');
    }
  }

  async uploadS3(buffer, originalName, mimeType) {
    try {
      const AWS = require('aws-sdk');
      const s3 = new AWS.S3({
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
        region: process.env.AWS_REGION || 'us-east-1'
      });

      const fileId = uuidv4();
      const ext = path.extname(originalName);
      const fileName = `${fileId}${ext}`;
      const key = `ilead-ams-documents/${fileName}`;

      const params = {
        Bucket: process.env.AWS_S3_BUCKET,
        Key: key,
        Body: buffer,
        ContentType: mimeType,
        ACL: 'private'
      };

      const result = await s3.upload(params).promise();
      console.log(`✅ File uploaded to S3: ${fileName}`);

      return {
        fileId,
        fileName,
        storagePath: key,
        storageUrl: result.Location,
        size: buffer.length
      };
    } catch (err) {
      console.error('S3 upload failed:', err);
      throw new Error('Failed to upload file to S3');
    }
  }

  async uploadDocument(buffer, originalName, mimeType, userId, organizationId, apprenticeId = null) {
    try {
      // Validate file
      this.validateFile(buffer, mimeType, originalName);

      // Upload to storage
      let storageResult;
      if (this.storageType === 's3') {
        storageResult = await this.uploadS3(buffer, originalName, mimeType);
      } else {
        storageResult = await this.uploadLocal(buffer, originalName);
      }

      // Save metadata to database
      const [document] = await db('documents')
        .insert({
          id: storageResult.fileId,
          filename: originalName,
          file_size: storageResult.size,
          mime_type: mimeType,
          storage_url: storageResult.storageUrl,
          uploaded_by: userId,
          organization_id: organizationId,
          apprentice_id: apprenticeId,
          created_at: new Date(),
          updated_at: new Date()
        })
        .returning('*');

      return {
        id: document.id,
        filename: document.filename,
        fileSize: document.file_size,
        mimeType: document.mime_type,
        url: document.storage_url,
        uploadedAt: document.created_at
      };
    } catch (err) {
      console.error('Document upload error:', err);
      throw err;
    }
  }

  async getDocument(documentId, organizationId) {
    try {
      const document = await db('documents')
        .where('id', documentId)
        .where('organization_id', organizationId)
        .first();

      if (!document) {
        throw new Error('Document not found');
      }

      return {
        id: document.id,
        filename: document.filename,
        fileSize: document.file_size,
        mimeType: document.mime_type,
        url: document.storage_url,
        uploadedAt: document.created_at
      };
    } catch (err) {
      console.error('Get document error:', err);
      throw err;
    }
  }

  async listDocuments(organizationId, filters = {}) {
    try {
      let query = db('documents').where('organization_id', organizationId);

      if (filters.apprenticeId) {
        query = query.where('apprentice_id', filters.apprenticeId);
      }

      const limit = Math.min(parseInt(filters.limit) || 50, 100);
      const offset = parseInt(filters.offset) || 0;

      return query
        .orderBy('created_at', 'desc')
        .limit(limit)
        .offset(offset);
    } catch (err) {
      console.error('List documents error:', err);
      throw err;
    }
  }

  async deleteDocument(documentId, organizationId) {
    try {
      const document = await db('documents')
        .where('id', documentId)
        .where('organization_id', organizationId)
        .first();

      if (!document) {
        throw new Error('Document not found');
      }

      // Delete from storage
      if (this.storageType === 'local') {
        try {
          const filePath = path.join(this.uploadDir, path.basename(document.storage_url));
          await fs.unlink(filePath);
          console.log(`✅ File deleted locally: ${filePath}`);
        } catch (err) {
          console.warn('Failed to delete local file:', err);
        }
      } else if (this.storageType === 's3') {
        try {
          const AWS = require('aws-sdk');
          const s3 = new AWS.S3();
          await s3.deleteObject({
            Bucket: process.env.AWS_S3_BUCKET,
            Key: document.storage_url
          }).promise();
          console.log(`✅ File deleted from S3`);
        } catch (err) {
          console.warn('Failed to delete S3 file:', err);
        }
      }

      // Delete from database
      await db('documents').where('id', documentId).del();
      console.log(`✅ Document record deleted: ${documentId}`);

      return true;
    } catch (err) {
      console.error('Delete document error:', err);
      throw err;
    }
  }
}

module.exports = new DocumentService();
