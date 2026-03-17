const request = require('supertest');
const app = require('../server');
const { createTestOrganization, createTestUser, createTestProgram } = require('./setup');
const { generateAccessToken } = require('../utils/jwt');

describe('Programs', () => {
  let testOrg;
  let adminUser;
  let apprenticeUser;
  let adminToken;
  let apprenticeToken;
  let testProgram;

  beforeAll(async () => {
    testOrg = await createTestOrganization();
    adminUser = await createTestUser(testOrg.id, 'administrator', 'admin@example.com');
    apprenticeUser = await createTestUser(testOrg.id, 'apprentice', 'apprentice@example.com');
    adminToken = generateAccessToken(adminUser);
    apprenticeToken = generateAccessToken(apprenticeUser);
    testProgram = await createTestProgram(testOrg.id);
  });

  describe('GET /api/programs', () => {
    it('should list programs for organization', async () => {
      const res = await request(app)
        .get('/api/programs')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
      expect(res.body.data.length).toBeGreaterThan(0);
    });

    it('should include program details', async () => {
      const res = await request(app)
        .get('/api/programs')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      const program = res.body.data[0];
      expect(program).toHaveProperty('id');
      expect(program).toHaveProperty('name');
      expect(program).toHaveProperty('code');
      expect(program).toHaveProperty('targetOjtHours');
      expect(program).toHaveProperty('targetRtiHours');
    });

    it('should be accessible to all authenticated users', async () => {
      const res = await request(app)
        .get('/api/programs')
        .set('Authorization', `Bearer ${apprenticeToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
    });
  });

  describe('GET /api/programs/:id', () => {
    it('should get program details', async () => {
      const res = await request(app)
        .get(`/api/programs/${testProgram.id}`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.program.id).toBe(testProgram.id);
      expect(res.body.program.name).toBe(testProgram.name);
      expect(res.body.program.code).toBe(testProgram.code);
    });

    it('should include OJT and RTI targets', async () => {
      const res = await request(app)
        .get(`/api/programs/${testProgram.id}`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.program.targetOjtHours).toBeDefined();
      expect(res.body.program.targetRtiHours).toBeDefined();
    });

    it('should return 404 for nonexistent program', async () => {
      const res = await request(app)
        .get('/api/programs/nonexistent-id')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(404);
      expect(res.body.error).toBe('Not Found');
    });

    it('should prevent cross-organization access', async () => {
      const otherOrg = await createTestOrganization();
      const otherProgram = await createTestProgram(otherOrg.id);

      const res = await request(app)
        .get(`/api/programs/${otherProgram.id}`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(403);
      expect(res.body.error).toBe('Forbidden');
    });
  });

  describe('POST /api/programs', () => {
    it('should create program as super admin', async () => {
      const superAdminUser = await createTestUser(testOrg.id, 'super_admin', 'superadmin@example.com');
      const superAdminToken = generateAccessToken(superAdminUser);

      const res = await request(app)
        .post('/api/programs')
        .set('Authorization', `Bearer ${superAdminToken}`)
        .send({
          name: 'New Program',
          code: 'NEW-REG',
          type: 'registeredApprenticeship',
          targetOjtHours: 2000,
          targetRtiHours: 184,
          partner: 'New Partner'
        });

      expect(res.status).toBe(201);
      expect(res.body.message).toContain('created');
      expect(res.body.program.name).toBe('New Program');
      expect(res.body.program.code).toBe('NEW-REG');
    });

    it('should prevent admin from creating program', async () => {
      const res = await request(app)
        .post('/api/programs')
        .set('Authorization', `Bearer ${adminToken}`)
        .send({
          name: 'New Program',
          code: 'NEW-REG',
          type: 'registeredApprenticeship',
          targetOjtHours: 2000,
          targetRtiHours: 184
        });

      expect(res.status).toBe(403);
      expect(res.body.error).toBe('Forbidden');
    });

    it('should prevent apprentice from creating program', async () => {
      const res = await request(app)
        .post('/api/programs')
        .set('Authorization', `Bearer ${apprenticeToken}`)
        .send({
          name: 'New Program',
          code: 'NEW-REG'
        });

      expect(res.status).toBe(403);
    });
  });

  describe('DELETE /api/programs/:id', () => {
    it('should delete program as super admin', async () => {
      const superAdminUser = await createTestUser(testOrg.id, 'super_admin', 'superadmin2@example.com');
      const superAdminToken = generateAccessToken(superAdminUser);
      const programToDelete = await createTestProgram(testOrg.id);

      const res = await request(app)
        .delete(`/api/programs/${programToDelete.id}`)
        .set('Authorization', `Bearer ${superAdminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.message).toContain('deleted');
    });

    it('should prevent admin from deleting program', async () => {
      const res = await request(app)
        .delete(`/api/programs/${testProgram.id}`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(403);
    });

    it('should return 404 for nonexistent program', async () => {
      const superAdminUser = await createTestUser(testOrg.id, 'super_admin', 'superadmin3@example.com');
      const superAdminToken = generateAccessToken(superAdminUser);

      const res = await request(app)
        .delete('/api/programs/nonexistent-id')
        .set('Authorization', `Bearer ${superAdminToken}`);

      expect(res.status).toBe(404);
    });
  });
});
