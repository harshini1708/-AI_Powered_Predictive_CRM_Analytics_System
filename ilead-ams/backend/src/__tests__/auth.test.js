const request = require('supertest');
const app = require('../server');
const { createTestOrganization, createTestUser } = require('./setup');
const { hashPassword } = require('../utils/password');

describe('Authentication', () => {
  let testOrg;

  beforeAll(async () => {
    testOrg = await createTestOrganization();
  });

  describe('POST /api/auth/register', () => {
    it('should register a new user', async () => {
      const res = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'newuser@example.com',
          password: 'SecurePassword123',
          firstName: 'John',
          lastName: 'Doe',
          phone: '6105551234',
          role: 'apprentice',
          programCode: 'RBT-REG'
        });

      expect(res.status).toBe(201);
      expect(res.body.message).toContain('created');
      expect(res.body.user.email).toBe('newuser@example.com');
      expect(res.body.user.status).toBe('pending_approval');
      expect(res.body.accessToken).toBeDefined();
    });

    it('should reject duplicate email', async () => {
      const email = 'duplicate@example.com';

      // First registration
      await request(app)
        .post('/api/auth/register')
        .send({
          email,
          password: 'SecurePassword123',
          firstName: 'First',
          lastName: 'User',
          role: 'apprentice'
        });

      // Second registration with same email
      const res = await request(app)
        .post('/api/auth/register')
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

    it('should not allow super_admin registration', async () => {
      const res = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'admin@example.com',
          password: 'SecurePassword123',
          firstName: 'Admin',
          lastName: 'User',
          role: 'super_admin'
        });

      expect(res.status).toBe(400);
      expect(res.body.error || res.body.message).toBeTruthy();
    });

    it('should validate password length', async () => {
      const res = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'weak@example.com',
          password: 'short',
          firstName: 'Test',
          lastName: 'User',
          role: 'apprentice'
        });

      expect(res.status).toBe(400);
    });
  });

  describe('POST /api/auth/login', () => {
    let testUser;
    const testPassword = 'TestPassword123';

    beforeAll(async () => {
      const passwordHash = await hashPassword(testPassword);
      const [user] = await require('../config/database')('users')
        .insert({
          id: require('uuid').v4(),
          email: 'logintest@example.com',
          password_hash: passwordHash,
          first_name: 'Login',
          last_name: 'Test',
          role: 'apprentice',
          organization_id: testOrg.id,
          status: 'active',
          created_at: new Date(),
          updated_at: new Date()
        })
        .returning('*');
      testUser = user;
    });

    it('should login with correct credentials', async () => {
      const res = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'logintest@example.com',
          password: testPassword
        });

      expect(res.status).toBe(200);
      expect(res.body.user.email).toBe('logintest@example.com');
      expect(res.body.accessToken).toBeDefined();
      expect(res.body.refreshToken).toBeDefined();
    });

    it('should reject incorrect password', async () => {
      const res = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'logintest@example.com',
          password: 'WrongPassword'
        });

      expect(res.status).toBe(401);
      expect(res.body.error).toBe('Unauthorized');
    });

    it('should reject nonexistent email', async () => {
      const res = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'nonexistent@example.com',
          password: testPassword
        });

      expect(res.status).toBe(401);
    });

    it('should reject pending approval accounts', async () => {
      const passwordHash = await hashPassword('TestPassword123');
      await require('../config/database')('users')
        .insert({
          id: require('uuid').v4(),
          email: 'pending@example.com',
          password_hash: passwordHash,
          first_name: 'Pending',
          last_name: 'User',
          role: 'apprentice',
          organization_id: testOrg.id,
          status: 'pending_approval',
          created_at: new Date(),
          updated_at: new Date()
        });

      const res = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'pending@example.com',
          password: 'TestPassword123'
        });

      expect(res.status).toBe(403);
      expect(res.body.code).toBe('ACCOUNT_PENDING_APPROVAL');
    });
  });

  describe('GET /api/auth/me', () => {
    it('should return current user with valid token', async () => {
      const user = await createTestUser(testOrg.id, 'apprentice', 'authme@example.com');
      const { generateAccessToken } = require('../utils/jwt');
      const token = generateAccessToken(user);

      const res = await request(app)
        .get('/api/auth/me')
        .set('Authorization', `Bearer ${token}`);

      expect(res.status).toBe(200);
      expect(res.body.user.email).toBe('authme@example.com');
    });

    it('should reject request without token', async () => {
      const res = await request(app)
        .get('/api/auth/me');

      expect(res.status).toBe(401);
    });

    it('should reject invalid token', async () => {
      const res = await request(app)
        .get('/api/auth/me')
        .set('Authorization', 'Bearer invalid-token');

      expect(res.status).toBe(401);
    });
  });
});
