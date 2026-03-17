const knex = require('knex');
const path = require('path');

const environment = process.env.NODE_ENV || 'development';
const config = {
  development: {
    client: 'pg',
    connection: {
      host: process.env.DB_HOST || 'localhost',
      port: process.env.DB_PORT || 5432,
      user: process.env.DB_USER || 'postgres',
      password: process.env.DB_PASSWORD || 'postgres',
      database: process.env.DB_NAME || 'ilead_ams',
      ssl: process.env.DB_SSL === 'true' ? { rejectUnauthorized: false } : false
    },
    migrations: {
      directory: path.join(__dirname, '../migrations')
    },
    seeds: {
      directory: path.join(__dirname, '../seeds')
    },
    pool: { min: 2, max: 10 },
    debug: true
  },
  production: {
    client: 'pg',
    connection: {
      host: process.env.DB_HOST,
      port: process.env.DB_PORT || 5432,
      user: process.env.DB_USER,
      password: process.env.DB_PASSWORD,
      database: process.env.DB_NAME,
      ssl: { rejectUnauthorized: false }
    },
    migrations: {
      directory: path.join(__dirname, '../migrations')
    },
    seeds: {
      directory: path.join(__dirname, '../seeds')
    },
    pool: { min: 5, max: 20 },
    debug: false
  },
  test: {
    client: 'sqlite3',
    connection: ':memory:',
    migrations: {
      directory: path.join(__dirname, '../migrations')
    },
    seeds: {
      directory: path.join(__dirname, '../seeds')
    },
    useNullAsDefault: true
  }
};

const db = knex(config[environment]);

module.exports = db;
