const express = require('express');
const router = express.Router();
const multer = require('multer');
const { authenticate } = require('../middleware/auth');
const documentController = require('../controllers/documentController');

// Configure multer for file uploads (memory storage, validation done in service)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 52 * 1024 * 1024 // 52MB limit (soft limit, checked in service)
  }
});

// Routes
router.post('/', authenticate, upload.single('file'), documentController.upload);
router.get('/', authenticate, documentController.list);
router.delete('/:id', authenticate, documentController.delete);

module.exports = router;
