const requestLogger = (req, res, next) => {
  const startTime = Date.now();

  // Log request
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`, {
    ip: req.ip,
    userAgent: req.get('user-agent')
  });

  // Override res.json to log response
  const originalJson = res.json;
  res.json = function(data) {
    const duration = Date.now() - startTime;
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.path} - ${res.statusCode} (${duration}ms)`);

    return originalJson.call(this, data);
  };

  next();
};

module.exports = {
  requestLogger
};
