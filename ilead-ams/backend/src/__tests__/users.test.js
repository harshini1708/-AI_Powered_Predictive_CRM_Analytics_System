const request = require('supertest');
const app = require('../server');
const { createTestOrganization, createTestUser } = require('./setup');
const { generateAccessToken } = require('../utils/jwt');

describe('User Management', () => {
  let testOrg;
  let adminUser;
  let superAdminToken;
  let adminToken;

  beforeAll(async () => {
    testOrg = await createTestOrganization();
    adminUser = await createTestUser(testOrg.id, 'administrator', 'admin@example.com');
    superAdminUser = await createTestUser(testOrg.id, 'super_admin', 'superadmin@example.com');

    superAdminToken = generateAccessToken(superAdminUser);
    adminToken = generateAccessToken(adminUser);
  });

  describe('POST /api/users', () => {
    it('should create a new user as super admin', async () => {
      const res = await request(app)
        .post('/api/users')
        .set('Authorization', `Bearer ${superAdminToken}`)
        .send({
          email: 'newuser@example.com',
          password: 'SecurePassword123',
          firstName: 'New',
          lastName: 'User',
          phone: '6105551234',
          role: 'supervisor'
        });

      expect(res.status).toBe(201);
      expect(res.body.message).toContain('created');
      expect(res.body.user.email).toBe('newuser@example.com');
      expect(res.body.user.role).toBe('supervisor');
    });

    it('should prevent super_admin role creation', async () => {
      const res = await request(app)
        .post('/api/users')
        .set('Authorization', `Bearer ${superAdminToken}`)
        .send({
          email: 'admin2@example.com',
          password: 'SecurePassword123',
          firstName: 'Admin',
          lastName: 'User',
          role: 'super_admin'
        });

      expect(res.status).toBe(400);
      expect(res.body.error).toBe('Bad Request');
    });

    it('should reject duplicate email', async () => {
      const email = 'duplicate@example.com';

      // Create first user
      await request(app)
        .post('/api/users')
        .set('Authorization', `Bearer ${superAdminToken}`)
        .send({
          email,
          password: 'SecurePassword123',
          firstName: 'First',
          lastName: 'User',
          role: 'apprentice'
        });

      // Try to create second user with same email
      const res = await request(app)
        .post('/api/users')
        .set('Authorization', `Bearer ${superAdminToken}`)
        .send({
          email,
          password: 'SecurePassword123',
          firstName: 'Second',
          lastName: 'User',
          role: 'apprentice'
        });

      expect(res.status).toBe(409);
      expect(res.body.error).toBe('Conflict');
    });
  });

  describe('GET /api/users', () => {
    it('should list users in organization', async () => {
      const res = await request(app)
        .get('/api/users')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
      expect(res.body.data.length).toBeGreaterThan(0);
    });

    it('should filter users by role', async () => {
      const res = await request(app)
        .get('/api/users?role=apprentice')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
    });

    it('should filter users by status', async () => {
      const res = await request(app)
        .get('/api/users?status=active')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
    });
  });

  describe('GET /api/users/:id', () => {
    it('should get user details', async () => {
      const res = await request(app)
        .get(`/api/users/${adminUser.id}`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.user.id).toBe(adminUser.id);
      expect(res.body.user.email).toBe(adminUser.email);
    });

    it('should return 404 for nonexistent user', async () => {
      const res = await request(app)
        .get('/api/users/nonexistent-id')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(404);
      expect(res.body.error).toBe('Not Found');
    });

    it('should prevent cross-organization access', async () => {
      const otherOrg = await createTestOrganization();
      const otherUser = await createTestUser(otherOrg.id, 'apprentice', 'other@example.com');

      const res = await request(app)
        .get(`/api/users/${otherUser.id}`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(403);
      expect(res.body.error).toBe('Forbidden');
    });
  });

  describe('PUT /api/users/:id', () => {
    it('should update user profile', async () => {
      const user = await createTestUser(testOrg.id, 'apprentice', 'update@example.com');

      const res = await request(app)
        .put(`/api/users/${user.id}`)
        .set('Authorization', `Bearer ${generateAccessToken(user)}`)
        .send({
          firstName: 'Updated',
          lastName: 'Name',
          phone: '5551234567'
        });

      expect(res.status).toBe(200);
      expect(res.body.message).toContain('updated');
      expect(res.body.user.firstName).toBe('Updated');
    });

    it('should allow admin to update other users', async () => {
      const user = await createTestUser(testOrg.id, 'apprentice', 'adminupdate@example.com');

      const res = await request(app)
        .put(`/api/users/${user.id}`)
        .set('Authorization', `Bearer ${adminToken}`)
        .send({
          firstName: 'AdminUpdated',
          lastName: 'User'
        });

      expect(res.status).toBe(200);
      expect(res.body.user.firstName).toBe('AdminUpdated');
    });

    it('should prevent users from changing their own role', async () => {
      const user = await createTestUser(testOrg.id, 'apprentice', 'rolechange@example.com');
      const token = generateAccessToken(user);

      const res = await request(app)
        .put(`/api/users/${user.id}`)
        .set('Authorization', `Bearer ${token}`)
        .send({
          role: 'administrator'
        });

      expect(res.status).toBe(400);
      expect(res.body.error).toBe('Bad Request');
    });

    it('should allow super admin to change roles', async () => {
      const user = await createTestUser(testOrg.id, 'apprentice', 'roleupdateadmin@example.com');

      const res = await request(app)
        .put(`/api/users/${user.id}`)
        .set('Authorization', `Bearer ${superAdminToken}`)
        .send({
          role: 'supervisor'
        });

      expect(res.status).toBe(200);
      expect(res.body.user.role).toBe('supervisor');
    });

    it('should return 404 for nonexistent user', async () => {
      const res = await request(app)
        .put('/api/users/nonexistent-id')
        .set('Authorization', `Bearer ${adminToken}`)
        .send({
          firstName: 'Updated'
        });

      expect(res.status).toBe(404);
    });
  });

  describe('DELETE /api/users/:id', () => {
    it('should delete user as super admin', async () => {
      const user = await createTestUser(testOrg.id, 'apprentice', 'delete@example.com');

      const res = await request(app)
        .delete(`/api/users/${user.id}`)
        .set('Authorization', `Bearer ${superAdminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.message).toContain('deleted');
    });

    it('should prevent deleting super admin', async () => {
      const res = await request(app)
        .delete(`/api/users/${superAdminUser.id}`)
        .set('Authorization', `Bearer ${superAdminToken}`);

      expect(res.status).toBe(400);
      expect(res.body.error).toBe('Bad Request');
    });

    it('should return 404 for nonexistent user', async () => {
      const res = await request(app)
        .delete('/api/users/nonexistent-id')
        .set('Authorization', `Bearer ${superAdminToken}`);

      expect(res.status).toBe(404);
    });
  });

  describe('GET /api/users/pending-approvals', () => {
    it('should list pending approval users', async () => {
      const res = await request(app)
        .get('/api/users/pending-approvals')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(Array.isArray(res.body.data)).toBe(true);
      expect(res.body).toHaveProperty('count');
    });
  });

  describe('POST /api/users/:id/approve', () => {
    it('should approve pending user', async () => {
      const user = await createTestUser(testOrg.id, 'apprentice', 'pending@example.com');
      // Manually set status to pending_approval
      const db = require('../config/database');
      await db('users').where('id', user.id).update({ status: 'pending_approval' });

      const res = await request(app)
        .post(`/api/users/${user.id}/approve`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.message).toContain('approved');
      expect(res.body.user.status).toBe('active');
    });

    it('should reject already approved user', async () => {
      const user = await createTestUser(testOrg.id, 'apprentice', 'alreadyapproved@example.com');

      const res = await request(app)
        .post(`/api/users/${user.id}/approve`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(400);
      expect(res.body.error).toBe('Bad Request');
    });

    it('should return 404 for nonexistent user', async () => {
      const res = await request(app)
        .post('/api/users/nonexistent-id/approve')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(404);
    });
  });

  describe('POST /api/users/:id/deactivate', () => {
    it('should deactivate user', async () => {
      const user = await createTestUser(testOrg.id, 'apprentice', 'deactivate@example.com');

      const res = await request(app)
        .post(`/api/users/${user.id}/deactivate`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(200);
      expect(res.body.message).toContain('deactivated');
      expect(res.body.user.status).toBe('inactive');
    });

    it('should return 404 for nonexistent user', async () => {
      const res = await request(app)
        .post('/api/users/nonexistent-id/deactivate')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(res.status).toBe(404);
    });
  });
});
