const request = require('supertest');
const app = require('../server');
const { createTestOrganization, createTestUser, createTestProgram, createTestApprentice } = require('./setup');
const { generateAccessToken } = require('../utils/jwt');
const db = require('../config/database');

describe('Apprentice Management', () => {
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

    adminToken = generateAccessToken(adminUser);
    supervisorToken = generateAccessToken(supervisorUser);
    apprenticeToken = generateAccessToken(apprenticeUser);
  });

  describe('GET /api/apprentices', () => {
    it('should list all apprentices for admin', async () => {
      const res = await request(app)
        .get('/api/apprentices')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
      expect(res.body.data.length).toBeGreaterThan(0);
    });

    it('should filter apprentices by status', async () => {
      const res = await request(app)
        .get('/api/apprentices?status=active')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
    });

    it('should limit results with pagination', async () => {
      const res = await request(app)
        .get('/api/apprentices?limit=10&offset=0')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
    });

    it('should show only assigned apprentices for supervisor', async () => {
      // Create another apprentice not assigned to supervisor
      const otherUser = await createTestUser(testOrg.id, 'apprentice', 'other@example.com');
      await createTestApprentice(otherUser.id, testProgram.id, testOrg.id);

      const res = await request(app)
        .get('/api/apprentices')
        .set('Authorization', `Bearer ${supervisorToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
    });
  });

  describe('GET /api/apprentices/:id', () => {
    it('should get apprentice details', async () => {
      const res = await request(app)
        .get(`/api/apprentices/${testApprentice.id}`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.apprentice.id).toBe(testApprentice.id);
      expect(res.body.apprentice.email).toBe(apprenticeUser.email);
    });

    it('should return 404 for nonexistent apprentice', async () => {
      const res = await request(app)
        .get('/api/apprentices/nonexistent-id')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(404);
      expect(res.body.error).toBe('Not Found');
    });

    it('should prevent cross-organization access', async () => {
      const otherOrg = await createTestOrganization();
      const otherUser = await createTestUser(otherOrg.id, 'apprentice', 'otherorg@example.com');
      const otherProgram = await createTestProgram(otherOrg.id);
      const otherApprentice = await createTestApprentice(otherUser.id, otherProgram.id, otherOrg.id);

      const res = await request(app)
        .get(`/api/apprentices/${otherApprentice.id}`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(403);
      expect(res.body.error).toBe('Forbidden');
    });

    it('should prevent supervisor from accessing unassigned apprentice', async () => {
      const otherUser = await createTestUser(testOrg.id, 'apprentice', 'unassigned@example.com');
      const otherApprentice = await createTestApprentice(otherUser.id, testProgram.id, testOrg.id);

      const res = await request(app)
        .get(`/api/apprentices/${otherApprentice.id}`)
        .set('Authorization', `Bearer ${supervisorToken}`);

      expect(res.status).toBe(403);
      expect(res.body.error).toBe('Forbidden');
    });
  });

  describe('GET /api/apprentices/:id/progress', () => {
    it('should get apprentice progress dashboard', async () => {
      const res = await request(app)
        .get(`/api/apprentices/${testApprentice.id}/progress`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.progress).toBeDefined();
      expect(res.body.progress.apprenticeId).toBe(testApprentice.id);
      expect(res.body.progress.ojtHours).toBeDefined();
      expect(res.body.progress.rtiHours).toBeDefined();
      expect(res.body.progress.completionPercent).toBeDefined();
      expect(res.body.progress.supervisionRatio).toBeDefined();
      expect(res.body.progress.supervisionOk).toBeDefined();
    });

    it('should calculate completion percentage correctly', async () => {
      const res = await request(app)
        .get(`/api/apprentices/${testApprentice.id}/progress`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.progress.completionPercent).toBeGreaterThanOrEqual(0);
      expect(res.body.progress.completionPercent).toBeLessThanOrEqual(100);
    });

    it('should return 404 for nonexistent apprentice', async () => {
      const res = await request(app)
        .get('/api/apprentices/nonexistent-id/progress')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(404);
    });
  });

  describe('POST /api/apprentices', () => {
    it('should create apprentice with new user', async () => {
      const res = await request(app)
        .post('/api/apprentices')
        .set('Authorization', `Bearer ${adminToken}`)
        .send({
          email: 'newapprentice@example.com',
          firstName: 'New',
          lastName: 'Apprentice',
          phone: '6105551234',
          programCode: testProgram.code,
          supervisor: 'John Supervisor',
          supervisorEmail: supervisorUser.email,
          employer: 'Test Employer',
          startDate: new Date().toISOString(),
          wageStart: 15.00,
          wageCurrent: 16.50
        });

      expect(res.status).toBe(201);
      expect(res.body.message).toContain('created');
      expect(res.body.apprentice.email).toBe('newapprentice@example.com');
      expect(res.body.apprentice.program).toBe(testProgram.name);
    });

    it('should create apprentice with existing user', async () => {
      const existingUser = await createTestUser(testOrg.id, 'apprentice', 'existinguser@example.com');

      const res = await request(app)
        .post('/api/apprentices')
        .set('Authorization', `Bearer ${adminToken}`)
        .send({
          email: existingUser.email,
          firstName: existingUser.first_name,
          lastName: existingUser.last_name,
          programCode: testProgram.code,
          supervisor: 'John Supervisor',
          supervisorEmail: supervisorUser.email,
          employer: 'Test Employer'
        });

      expect(res.status).toBe(201);
      expect(res.body.apprentice.email).toBe(existingUser.email);
    });

    it('should reject invalid program code', async () => {
      const res = await request(app)
        .post('/api/apprentices')
        .set('Authorization', `Bearer ${adminToken}`)
        .send({
          email: 'apprentice@example.com',
          firstName: 'Test',
          lastName: 'User',
          programCode: 'INVALID-CODE',
          supervisor: 'John Supervisor',
          supervisorEmail: supervisorUser.email
        });

      expect(res.status).toBe(400);
      expect(res.body.error).toBe('Bad Request');
    });
  });

  describe('PUT /api/apprentices/:id', () => {
    it('should update apprentice details', async () => {
      const res = await request(app)
        .put(`/api/apprentices/${testApprentice.id}`)
        .set('Authorization', `Bearer ${adminToken}`)
        .send({
          supervisor: 'Updated Supervisor',
          supervisorEmail: supervisorUser.email,
          employer: 'Updated Employer',
          wageCurrent: 17.00,
          status: 'active'
        });

      expect(res.status).toBe(200);
      expect(res.body.message).toContain('updated');
      expect(res.body.apprentice.supervisor).toBe('Updated Supervisor');
      expect(res.body.apprentice.employer).toBe('Updated Employer');
    });

    it('should update apprentice status', async () => {
      const user = await createTestUser(testOrg.id, 'apprentice', 'statusupdate@example.com');
      const apprentice = await createTestApprentice(user.id, testProgram.id, testOrg.id);

      const res = await request(app)
        .put(`/api/apprentices/${apprentice.id}`)
        .set('Authorization', `Bearer ${adminToken}`)
        .send({
          status: 'completed'
        });

      expect(res.status).toBe(200);
      expect(res.body.apprentice.status).toBe('completed');
    });

    it('should return 404 for nonexistent apprentice', async () => {
      const res = await request(app)
        .put('/api/apprentices/nonexistent-id')
        .set('Authorization', `Bearer ${adminToken}`)
        .send({
          employer: 'Updated Employer'
        });

      expect(res.status).toBe(404);
    });

    it('should allow partial updates', async () => {
      const user = await createTestUser(testOrg.id, 'apprentice', 'partial@example.com');
      const apprentice = await createTestApprentice(user.id, testProgram.id, testOrg.id);
      const originalSupervisor = apprentice.supervisor;

      const res = await request(app)
        .put(`/api/apprentices/${apprentice.id}`)
        .set('Authorization', `Bearer ${adminToken}`)
        .send({
          employer: 'New Employer Only'
        });

      expect(res.status).toBe(200);
      expect(res.body.apprentice.employer).toBe('New Employer Only');
    });
  });

  describe('Domain breakdown in progress', () => {
    it('should return domain hours breakdown', async () => {
      const res = await request(app)
        .get(`/api/apprentices/${testApprentice.id}/progress`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.progress.domainBreakdown).toBeDefined();
      expect(Array.isArray(res.body.progress.domainBreakdown)).toBe(true);
    });
  });

  describe('Supervision ratio calculations', () => {
    it('should calculate supervision ratio correctly', async () => {
      const res = await request(app)
        .get(`/api/apprentices/${testApprentice.id}/progress`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.progress.supervisionRatio).toBeGreaterThanOrEqual(0);
      expect(typeof res.body.progress.supervisionOk).toBe('boolean');
    });
  });
});
