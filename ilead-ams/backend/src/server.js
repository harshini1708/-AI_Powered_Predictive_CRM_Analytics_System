require('dotenv').config();
require('express-async-errors');
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const bodyParser = require('body-parser');
const swaggerUi = require('swagger-ui-express');
const YAML = require('yamljs');

// Import middleware
const { errorHandler } = require('./middleware/errorHandler');
const { requestLogger } = require('./middleware/requestLogger');

// Import routes
const authRoutes = require('./routes/auth');
const userRoutes = require('./routes/users');
const apprenticeRoutes = require('./routes/apprentices');
const hourLogRoutes = require('./routes/hourLogs');
const programRoutes = require('./routes/programs');
const organizationRoutes = require('./routes/organizations');
const analyticsRoutes = require('./routes/analytics');
const webhookRoutes = require('./routes/webhooks');
const documentRoutes = require('./routes/documents');

const app = express();
const PORT = process.env.PORT || 3000;

// Trust proxy (important for production behind nginx/load balancer)
app.set('trust proxy', 1);

// Security Middleware
app.use(helmet());

// CORS Configuration
const corsOptions = {
  origin: (process.env.CORS_ALLOWED_ORIGINS || 'http://localhost:3000').split(','),
  credentials: true,
  optionsSuccessStatus: 200,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
};
app.use(cors(corsOptions));

// Rate Limiting (disabled in test mode)
const isTestMode = process.env.NODE_ENV === 'test';
const limiter = isTestMode ? (req, res, next) => next() : rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 900000, // 15 minutes
  max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100,
  message: 'Too many requests from this IP, please try again later.'
});
const authLimiter = isTestMode ? (req, res, next) => next() : rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // 5 requests per window
  skipSuccessfulRequests: true,
  message: 'Too many login attempts, please try again later.'
});

app.use(limiter);

// Body Parser Middleware
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));

// Request Logging Middleware
app.use(requestLogger);

// Health Check Endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    environment: process.env.NODE_ENV || 'development'
  });
});

// Swagger UI Documentation (Development)
if (process.env.NODE_ENV !== 'production') {
  try {
    const swaggerDocument = YAML.load('./openapi.yml');
    app.use('/docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument, {
      swaggerOptions: {
        urls: [
          {
            url: '/openapi.json',
            name: 'OpenAPI JSON'
          }
        ],
        persistAuthorization: true
      }
    }));

    // Serve OpenAPI spec as JSON
    app.get('/openapi.json', (req, res) => {
      res.json(swaggerDocument);
    });

    // Redirect /api-docs to /docs
    app.get('/api-docs', (req, res) => {
      res.redirect('/docs');
    });
  } catch (err) {
    console.warn('Warning: Could not load Swagger documentation:', err.message);
  }
}

// API Version Prefix
const apiVersion = '/api';

// Routes
app.use(`${apiVersion}/auth`, authLimiter, authRoutes);
app.use(`${apiVersion}/users`, userRoutes);
app.use(`${apiVersion}/apprentices`, apprenticeRoutes);
app.use(`${apiVersion}/hour-logs`, hourLogRoutes);
app.use(`${apiVersion}/programs`, programRoutes);
app.use(`${apiVersion}/organizations`, organizationRoutes);
app.use(`${apiVersion}/analytics`, analyticsRoutes);
app.use(`${apiVersion}/documents`, documentRoutes);
app.use(`${apiVersion}/webhooks`, webhookRoutes);

// 404 Handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Route ${req.method} ${req.path} not found`,
    timestamp: new Date().toISOString()
  });
});

// Global Error Handler
app.use(errorHandler);

// Start Server (only if not in test mode)
let server;
if (process.env.NODE_ENV !== 'test') {
  server = app.listen(PORT, () => {
    console.log(`
╔════════════════════════════════════════════════╗
║  I-LEAD AMS Backend Server                    ║
║  Version: 1.0.0                               ║
║  Environment: ${process.env.NODE_ENV || 'development'.padEnd(30)} ║
║  Port: ${PORT.toString().padEnd(39)} ║
║  Started: ${new Date().toISOString().padEnd(38)} ║
╚════════════════════════════════════════════════╝
    `);
  });

  // Graceful Shutdown
  process.on('SIGTERM', () => {
    console.log('SIGTERM signal received: closing HTTP server');
    server.close(() => {
      console.log('HTTP server closed');
      process.exit(0);
    });
  });

  process.on('SIGINT', () => {
    console.log('SIGINT signal received: closing HTTP server');
    server.close(() => {
      console.log('HTTP server closed');
      process.exit(0);
    });
  });
}

module.exports = app;
