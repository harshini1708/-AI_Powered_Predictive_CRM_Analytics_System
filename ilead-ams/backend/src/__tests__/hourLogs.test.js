const request = require('supertest');
const app = require('../server');
const { createTestOrganization, createTestUser, createTestProgram, createTestApprentice } = require('./setup');
const { generateAccessToken } = require('../utils/jwt');
const db = require('../config/database');

describe('Hour Logs & Approval', () => {
  let testOrg;
  let adminUser;
  let supervisorUser;
  let apprenticeUser;
  let testProgram;
  let testApprentice;
  let adminToken;
  let supervisorToken;
  let apprenticeToken;

  beforeAll(async () => {
    testOrg = await createTestOrganization();
    adminUser = await createTestUser(testOrg.id, 'administrator', 'admin@example.com');
    supervisorUser = await createTestUser(testOrg.id, 'supervisor', 'supervisor@example.com');
    apprenticeUser = await createTestUser(testOrg.id, 'apprentice', 'apprentice@example.com');
    testProgram = await createTestProgram(testOrg.id);
    testApprentice = await createTestApprentice(apprenticeUser.id, testProgram.id, testOrg.id);

    // Assign supervisor to apprentice
    await db('apprentices').where('id', testApprentice.id).update({
      supervisor: 'Test Supervisor',
      supervisor_email: supervisorUser.email
    });

    adminToken = generateAccessToken(adminUser);
    supervisorToken = generateAccessToken(supervisorUser);
    apprenticeToken = generateAccessToken(apprenticeUser);
  });

  describe('POST /api/hour-logs/submit', () => {
    it('should submit OJT hours', async () => {
      const res = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'dataCollection',
          ojtHours: 8,
          logDate: new Date().toISOString(),
          description: 'Data collection training'
        });

      expect(res.status).toBe(201);
      expect(res.body.message).toContain('submitted');
      expect(res.body.hourLog.ojtHours).toBe(8);
      expect(res.body.hourLog.status).toBe('pending');
    });

    it('should submit RTI hours', async () => {
      const res = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          rtiModule: 'ethics',
          rtiHours: 4,
          logDate: new Date().toISOString(),
          description: 'Ethics training'
        });

      expect(res.status).toBe(201);
      expect(res.body.hourLog.rtiHours).toBe(4);
    });

    it('should submit both OJT and RTI hours', async () => {
      const res = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'assessment',
          rtiModule: 'measurement',
          ojtHours: 6,
          rtiHours: 2,
          logDate: new Date().toISOString()
        });

      expect(res.status).toBe(201);
      expect(res.body.hourLog.ojtHours).toBe(6);
      expect(res.body.hourLog.rtiHours).toBe(2);
    });

    it('should reject submission without hours', async () => {
      const res = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          logDate: new Date().toISOString(),
          description: 'No hours'
        });

      expect(res.status).toBe(400);
      expect(res.body.error).toBe('Bad Request');
    });

    it('should reject invalid OJT domain', async () => {
      const res = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'invalidDomain',
          ojtHours: 8,
          logDate: new Date().toISOString()
        });

      expect(res.status).toBe(400);
      expect(res.body.message).toContain('OJT domain');
    });

    it('should reject invalid RTI module', async () => {
      const res = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          rtiModule: 'invalidModule',
          rtiHours: 4,
          logDate: new Date().toISOString()
        });

      expect(res.status).toBe(400);
      expect(res.body.message).toContain('RTI module');
    });

    it('should reject submission without apprentice profile', async () => {
      const loneUser = await createTestUser(testOrg.id, 'apprentice', 'noapprentice@example.com');
      const loneToken = generateAccessToken(loneUser);

      const res = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${loneToken}`)
        .send({
          ojtDomain: 'dataCollection',
          ojtHours: 8,
          logDate: new Date().toISOString()
        });

      expect(res.status).toBe(404);
    });
  });

  describe('GET /api/hour-logs', () => {
    it('should list apprentice logs for apprentice user', async () => {
      const res = await request(app)
        .get('/api/hour-logs')
        .set('Authorization', `Bearer ${apprenticeToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
      expect(res.body).toHaveProperty('count');
    });

    it('should list all logs for admin', async () => {
      const res = await request(app)
        .get('/api/hour-logs')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
    });

    it('should list pending logs for supervisor', async () => {
      const res = await request(app)
        .get('/api/hour-logs')
        .set('Authorization', `Bearer ${supervisorToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
    });

    it('should filter by status', async () => {
      const res = await request(app)
        .get('/api/hour-logs?status=pending')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
    });

    it('should support pagination', async () => {
      const res = await request(app)
        .get('/api/hour-logs?limit=10&offset=0')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
    });
  });

  describe('GET /api/hour-logs/:id', () => {
    it('should get hour log details', async () => {
      // Create a log first
      const submitRes = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'dataCollection',
          ojtHours: 8,
          logDate: new Date().toISOString()
        });

      const logId = submitRes.body.hourLog.id;

      const res = await request(app)
        .get(`/api/hour-logs/${logId}`)
        .set('Authorization', `Bearer ${apprenticeToken}`);

      expect(res.status).toBe(200);
      expect(res.body.hourLog.id).toBe(logId);
      expect(res.body.hourLog.ojtHours).toBe(8);
    });

    it('should return 404 for nonexistent log', async () => {
      const res = await request(app)
        .get('/api/hour-logs/nonexistent-id')
        .set('Authorization', `Bearer ${apprenticeToken}`);

      expect(res.status).toBe(404);
      expect(res.body.error).toBe('Not Found');
    });

    it('should prevent apprentice from viewing other apprentice logs', async () => {
      const otherUser = await createTestUser(testOrg.id, 'apprentice', 'otherapprentice@example.com');
      const otherToken = generateAccessToken(otherUser);
      const otherProgram = await createTestProgram(testOrg.id);
      const otherApprentice = await createTestApprentice(otherUser.id, otherProgram.id, testOrg.id);

      const submitRes = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${otherToken}`)
        .send({
          ojtDomain: 'dataCollection',
          ojtHours: 8,
          logDate: new Date().toISOString()
        });

      const res = await request(app)
        .get(`/api/hour-logs/${submitRes.body.hourLog.id}`)
        .set('Authorization', `Bearer ${apprenticeToken}`);

      expect(res.status).toBe(403);
    });
  });

  describe('POST /api/hour-logs/:id/approve', () => {
    it('should approve hours with rubric score >= 3', async () => {
      const submitRes = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'dataCollection',
          ojtHours: 8,
          logDate: new Date().toISOString()
        });

      const logId = submitRes.body.hourLog.id;

      const res = await request(app)
        .post(`/api/hour-logs/${logId}/approve`)
        .set('Authorization', `Bearer ${supervisorToken}`)
        .send({
          rubricScore: 4,
          rubricDomain: 'A',
          specificTask: 'A-1',
          rubricNotes: 'Good performance',
          supervisionMinutes: 30,
          supervisionType: 'direct'
        });

      expect(res.status).toBe(200);
      expect(res.body.message).toContain('approved');
      expect(res.body.hourLog.status).toBe('approved');
      expect(res.body.hourLog.rubricScore).toBe(4);
    });

    it('should qualify hours with score >= 3', async () => {
      const submitRes = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'assessment',
          ojtHours: 10,
          logDate: new Date().toISOString()
        });

      const logId = submitRes.body.hourLog.id;

      const res = await request(app)
        .post(`/api/hour-logs/${logId}/approve`)
        .set('Authorization', `Bearer ${supervisorToken}`)
        .send({
          rubricScore: 5,
          rubricDomain: 'B',
          specificTask: 'B-1',
          rubricNotes: 'Excellent',
          supervisionMinutes: 60,
          supervisionType: 'direct'
        });

      expect(res.status).toBe(200);
      expect(res.body.hourLog.qualifiedHours).toBe(10);
    });

    it('should not qualify hours with score < 3', async () => {
      const submitRes = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'skillAcquisition',
          ojtHours: 6,
          logDate: new Date().toISOString()
        });

      const logId = submitRes.body.hourLog.id;

      const res = await request(app)
        .post(`/api/hour-logs/${logId}/approve`)
        .set('Authorization', `Bearer ${supervisorToken}`)
        .send({
          rubricScore: 2,
          rubricDomain: 'C',
          specificTask: 'C-1',
          rubricNotes: 'Needs improvement',
          nextSteps: 'Complete refresher training',
          supervisionMinutes: 45,
          supervisionType: 'indirect'
        });

      expect(res.status).toBe(200);
      expect(res.body.hourLog.remediationRequired).toBe(true);
      expect(res.body.hourLog.qualifiedHours).toBeLessThan(6);
    });

    it('should reject approval with invalid rubric score', async () => {
      const submitRes = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'dataCollection',
          ojtHours: 8,
          logDate: new Date().toISOString()
        });

      const logId = submitRes.body.hourLog.id;

      const res = await request(app)
        .post(`/api/hour-logs/${logId}/approve`)
        .set('Authorization', `Bearer ${supervisorToken}`)
        .send({
          rubricScore: 10,
          rubricDomain: 'A',
          specificTask: 'A-1'
        });

      expect(res.status).toBe(400);
      expect(res.body.error).toBe('Bad Request');
    });

    it('should reject approval with invalid domain', async () => {
      const submitRes = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'dataCollection',
          ojtHours: 8,
          logDate: new Date().toISOString()
        });

      const logId = submitRes.body.hourLog.id;

      const res = await request(app)
        .post(`/api/hour-logs/${logId}/approve`)
        .set('Authorization', `Bearer ${supervisorToken}`)
        .send({
          rubricScore: 4,
          rubricDomain: 'Z',
          specificTask: 'Z-1'
        });

      expect(res.status).toBe(400);
      expect(res.body.message).toContain('RBT task domain');
    });

    it('should require remediation plan for score < 3', async () => {
      const submitRes = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'dataCollection',
          ojtHours: 8,
          logDate: new Date().toISOString()
        });

      const logId = submitRes.body.hourLog.id;

      const res = await request(app)
        .post(`/api/hour-logs/${logId}/approve`)
        .set('Authorization', `Bearer ${supervisorToken}`)
        .send({
          rubricScore: 2,
          rubricDomain: 'A',
          specificTask: 'A-1'
          // Missing nextSteps
        });

      expect(res.status).toBe(400);
      expect(res.body.message).toContain('Remediation');
    });

    it('should return 404 for nonexistent log', async () => {
      const res = await request(app)
        .post('/api/hour-logs/nonexistent-id/approve')
        .set('Authorization', `Bearer ${supervisorToken}`)
        .send({
          rubricScore: 4,
          rubricDomain: 'A',
          specificTask: 'A-1'
        });

      expect(res.status).toBe(404);
    });
  });

  describe('POST /api/hour-logs/:id/reject', () => {
    it('should reject hours with notes', async () => {
      const submitRes = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'dataCollection',
          ojtHours: 8,
          logDate: new Date().toISOString()
        });

      const logId = submitRes.body.hourLog.id;

      const res = await request(app)
        .post(`/api/hour-logs/${logId}/reject`)
        .set('Authorization', `Bearer ${supervisorToken}`)
        .send({
          rubricNotes: 'Hours do not align with activities'
        });

      expect(res.status).toBe(200);
      expect(res.body.message).toContain('rejected');
      expect(res.body.hourLog.status).toBe('rejected');
    });

    it('should reject with default message if no notes provided', async () => {
      const submitRes = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'assessment',
          ojtHours: 6,
          logDate: new Date().toISOString()
        });

      const logId = submitRes.body.hourLog.id;

      const res = await request(app)
        .post(`/api/hour-logs/${logId}/reject`)
        .set('Authorization', `Bearer ${supervisorToken}`)
        .send({});

      expect(res.status).toBe(200);
      expect(res.body.hourLog.status).toBe('rejected');
    });

    it('should return 404 for nonexistent log', async () => {
      const res = await request(app)
        .post('/api/hour-logs/nonexistent-id/reject')
        .set('Authorization', `Bearer ${supervisorToken}`)
        .send({
          rubricNotes: 'Invalid log'
        });

      expect(res.status).toBe(404);
    });
  });

  describe('Supervision minutes tracking', () => {
    it('should track supervision minutes on approval', async () => {
      const submitRes = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'dataCollection',
          ojtHours: 8,
          logDate: new Date().toISOString()
        });

      const logId = submitRes.body.hourLog.id;

      const approveRes = await request(app)
        .post(`/api/hour-logs/${logId}/approve`)
        .set('Authorization', `Bearer ${supervisorToken}`)
        .send({
          rubricScore: 4,
          rubricDomain: 'A',
          specificTask: 'A-1',
          supervisionMinutes: 45,
          supervisionType: 'direct'
        });

      expect(approveRes.status).toBe(200);
      expect(approveRes.body.hourLog.supervisionMinutes).toBe(45);
      expect(approveRes.body.apprenticeTotals.supervisionMinutes).toBeGreaterThanOrEqual(45);
    });
  });
});
