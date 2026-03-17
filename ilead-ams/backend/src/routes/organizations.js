const express = require('express');
const router = express.Router();
const { authenticate } = require('../middleware/auth');

router.get('/', authenticate, (req, res) => {
  res.json({ message: 'Organizations list endpoint' });
});

module.exports = router;
