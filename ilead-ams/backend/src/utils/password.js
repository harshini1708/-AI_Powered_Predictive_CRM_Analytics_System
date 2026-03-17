const bcrypt = require('bcryptjs');

const SALT_ROUNDS = 12;

async function hashPassword(password) {
  try {
    return await bcrypt.hash(password, SALT_ROUNDS);
  } catch (err) {
    throw new Error('Password hashing failed');
  }
}

async function verifyPassword(password, hash) {
  try {
    return await bcrypt.compare(password, hash);
  } catch (err) {
    return false;
  }
}

async function hashPin(pin) {
  try {
    return await bcrypt.hash(pin, SALT_ROUNDS);
  } catch (err) {
    throw new Error('PIN hashing failed');
  }
}

async function verifyPin(pin, hash) {
  try {
    return await bcrypt.compare(pin, hash);
  } catch (err) {
    return false;
  }
}

module.exports = {
  hashPassword,
  verifyPassword,
  hashPin,
  verifyPin
};
