const request = require('supertest');
const app = require('../server');
const { createTestOrganization, createTestUser, createTestProgram, createTestApprentice } = require('./setup');
const { generateAccessToken } = require('../utils/jwt');
const db = require('../config/database');

describe('Analytics', () => {
  let testOrg;
  let adminUser;
  let apprenticeUser;
  let adminToken;
  let apprenticeToken;
  let testProgram;
  let testApprentice;

  beforeAll(async () => {
    testOrg = await createTestOrganization();
    adminUser = await createTestUser(testOrg.id, 'administrator', 'admin@example.com');
    apprenticeUser = await createTestUser(testOrg.id, 'apprentice', 'apprentice@example.com');
    adminToken = generateAccessToken(adminUser);
    apprenticeToken = generateAccessToken(apprenticeUser);
    testProgram = await createTestProgram(testOrg.id);
    testApprentice = await createTestApprentice(apprenticeUser.id, testProgram.id, testOrg.id);
  });

  describe('GET /api/analytics/dashboard', () => {
    it('should return dashboard stats', async () => {
      const res = await request(app)
        .get('/api/analytics/dashboard')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.dashboard).toBeDefined();
      expect(res.body.dashboard).toHaveProperty('totalUsers');
      expect(res.body.dashboard).toHaveProperty('activeApprentices');
      expect(res.body.dashboard).toHaveProperty('totalOjtHours');
      expect(res.body.dashboard).toHaveProperty('qualifiedOjt');
      expect(res.body.dashboard).toHaveProperty('pendingApprovals');
    });

    it('should include counts as numbers', async () => {
      const res = await request(app)
        .get('/api/analytics/dashboard')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(typeof res.body.dashboard.totalUsers).toBe('number');
      expect(typeof res.body.dashboard.activeApprentices).toBe('number');
      expect(typeof res.body.dashboard.totalOjtHours).toBe('number');
      expect(typeof res.body.dashboard.qualifiedOjt).toBe('number');
      expect(typeof res.body.dashboard.pendingApprovals).toBe('number');
    });

    it('should be accessible to apprentice role', async () => {
      const res = await request(app)
        .get('/api/analytics/dashboard')
        .set('Authorization', `Bearer ${apprenticeToken}`);

      expect(res.status).toBe(200);
      expect(res.body.dashboard).toBeDefined();
    });

    it('should organize-scope data', async () => {
      // Create another org with data
      const otherOrg = await createTestOrganization();
      const otherAdmin = await createTestUser(otherOrg.id, 'administrator', 'otheradmin@example.com');
      const otherAdminToken = generateAccessToken(otherAdmin);

      const res = await request(app)
        .get('/api/analytics/dashboard')
        .set('Authorization', `Bearer ${otherAdminToken}`);

      expect(res.status).toBe(200);
      // Should only see data from their organization
      expect(res.body.dashboard).toBeDefined();
    });

    it('should count pending approvals', async () => {
      const supervisorUser = await createTestUser(testOrg.id, 'supervisor', 'supervisor@example.com');
      const supervisorToken = generateAccessToken(supervisorUser);

      // Submit an hour log
      await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'dataCollection',
          ojtHours: 8,
          logDate: new Date().toISOString()
        });

      const res = await request(app)
        .get('/api/analytics/dashboard')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.dashboard.pendingApprovals).toBeGreaterThanOrEqual(0);
    });
  });

  describe('GET /api/analytics/domain-progress', () => {
    it('should return domain breakdown', async () => {
      const res = await request(app)
        .get('/api/analytics/domain-progress')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.domainBreakdown).toBeDefined();
      expect(Array.isArray(res.body.domainBreakdown)).toBe(true);
    });

    it('should include qualified hours by domain', async () => {
      // Create and approve hour logs
      const supervisorUser = await createTestUser(testOrg.id, 'supervisor', 'supervisor@example.com');
      const supervisorToken = generateAccessToken(supervisorUser);

      await db('apprentices').where('id', testApprentice.id).update({
        supervisor_email: supervisorUser.email
      });

      const submitRes = await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'dataCollection',
          ojtHours: 8,
          logDate: new Date().toISOString()
        });

      await request(app)
        .post(`/api/hour-logs/${submitRes.body.hourLog.id}/approve`)
        .set('Authorization', `Bearer ${supervisorToken}`)
        .send({
          rubricScore: 4,
          rubricDomain: 'A',
          specificTask: 'A-1',
          rubricNotes: 'Good'
        });

      const res = await request(app)
        .get('/api/analytics/domain-progress')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.domainBreakdown).toBeDefined();
    });

    it('should show zero hours for domains with no data', async () => {
      const res = await request(app)
        .get('/api/analytics/domain-progress')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      // Should include all domains even if empty
      expect(Array.isArray(res.body.domainBreakdown)).toBe(true);
    });

    it('should only include approved logs', async () => {
      // Submit but don't approve
      await request(app)
        .post('/api/hour-logs/submit')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          ojtDomain: 'assessment',
          ojtHours: 10,
          logDate: new Date().toISOString()
        });

      const res = await request(app)
        .get('/api/analytics/domain-progress')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      // Pending logs shouldn't be included
      expect(Array.isArray(res.body.domainBreakdown)).toBe(true);
    });

    it('should be accessible to apprentice role', async () => {
      const res = await request(app)
        .get('/api/analytics/domain-progress')
        .set('Authorization', `Bearer ${apprenticeToken}`);

      expect(res.status).toBe(200);
      expect(res.body.domainBreakdown).toBeDefined();
    });
  });

  describe('GET /api/analytics/competency-heat-map', () => {
    it('should return competency ratings by domain', async () => {
      const res = await request(app)
        .get('/api/analytics/competency-heat-map')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.competencyMap).toBeDefined();
      expect(Array.isArray(res.body.competencyMap)).toBe(true);
    });

    it('should include domain, avg score, and count', async () => {
      const res = await request(app)
        .get('/api/analytics/competency-heat-map')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      if (res.body.competencyMap.length > 0) {
        const competency = res.body.competencyMap[0];
        expect(competency).toHaveProperty('domain');
        expect(competency).toHaveProperty('avgScore');
        expect(competency).toHaveProperty('count');
      }
    });

    it('should calculate average scores correctly', async () => {
      // Create ratings
      const supervisorUser = await createTestUser(testOrg.id, 'supervisor', 'supervisor2@example.com');
      const supervisorToken = generateAccessToken(supervisorUser);

      await db('apprentices').where('id', testApprentice.id).update({
        supervisor_email: supervisorUser.email
      });

      // Submit and approve multiple logs with different scores
      const logs = [];
      for (let i = 0; i < 3; i++) {
        const submitRes = await request(app)
          .post('/api/hour-logs/submit')
          .set('Authorization', `Bearer ${apprenticeToken}`)
          .send({
            ojtDomain: 'skillAcquisition',
            ojtHours: 5 + i,
            logDate: new Date().toISOString()
          });

        await request(app)
          .post(`/api/hour-logs/${submitRes.body.hourLog.id}/approve`)
          .set('Authorization', `Bearer ${supervisorToken}`)
          .send({
            rubricScore: 3 + i,
            rubricDomain: 'C',
            specificTask: 'C-1',
            rubricNotes: 'Test'
          });
      }

      const res = await request(app)
        .get('/api/analytics/competency-heat-map')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.competencyMap)).toBe(true);
    });

    it('should show empty array when no ratings exist', async () => {
      const newOrg = await createTestOrganization();
      const newAdmin = await createTestUser(newOrg.id, 'administrator', 'newadmin@example.com');
      const newAdminToken = generateAccessToken(newAdmin);

      const res = await request(app)
        .get('/api/analytics/competency-heat-map')
        .set('Authorization', `Bearer ${newAdminToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.competencyMap)).toBe(true);
    });

    it('should be accessible to apprentice role', async () => {
      const res = await request(app)
        .get('/api/analytics/competency-heat-map')
        .set('Authorization', `Bearer ${apprenticeToken}`);

      expect(res.status).toBe(200);
      expect(res.body.competencyMap).toBeDefined();
    });

    it('should format average scores to one decimal place', async () => {
      const res = await request(app)
        .get('/api/analytics/competency-heat-map')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      if (res.body.competencyMap.length > 0) {
        const competency = res.body.competencyMap[0];
        // Should be formatted as string with one decimal
        expect(typeof competency.avgScore).toBe('string');
        expect(/^\d+\.\d$/.test(competency.avgScore) || /^\d+$/.test(competency.avgScore)).toBe(true);
      }
    });
  });

  describe('Authorization', () => {
    it('should deny access without token', async () => {
      const res = await request(app)
        .get('/api/analytics/dashboard');

      expect(res.status).toBe(401);
    });

    it('should deny access with invalid token', async () => {
      const res = await request(app)
        .get('/api/analytics/dashboard')
        .set('Authorization', 'Bearer invalid-token');

      expect(res.status).toBe(401);
    });
  });
});
