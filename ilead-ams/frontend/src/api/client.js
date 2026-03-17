import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';
const API_TIMEOUT = parseInt(import.meta.env.VITE_API_TIMEOUT) || 30000;

// Create axios instance
const client = axios.create({
  baseURL: `${API_URL}/api`,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Token management
let accessToken = localStorage.getItem('accessToken');
let refreshToken = localStorage.getItem('refreshToken');

/**
 * Set tokens in storage and update headers
 */
export const setTokens = (access, refresh) => {
  accessToken = access;
  refreshToken = refresh;

  if (access) {
    localStorage.setItem('accessToken', access);
    client.defaults.headers.common['Authorization'] = `Bearer ${access}`;
  } else {
    localStorage.removeItem('accessToken');
    delete client.defaults.headers.common['Authorization'];
  }

  if (refresh) {
    localStorage.setItem('refreshToken', refresh);
  } else {
    localStorage.removeItem('refreshToken');
  }
};

/**
 * Clear tokens
 */
export const clearTokens = () => {
  setTokens(null, null);
};

/**
 * Get current access token
 */
export const getAccessToken = () => accessToken;

/**
 * Get current refresh token
 */
export const getRefreshToken = () => refreshToken;

// Set initial token from storage
if (accessToken) {
  client.defaults.headers.common['Authorization'] = `Bearer ${accessToken}`;
}

// Request interceptor - add auth token
client.interceptors.request.use(
  config => {
    if (accessToken && !config.headers.Authorization) {
      config.headers.Authorization = `Bearer ${accessToken}`;
    }
    return config;
  },
  error => Promise.reject(error)
);

// Response interceptor - handle 401 and refresh token
let isRefreshing = false;
let failedQueue = [];

const processQueue = (error, token = null) => {
  failedQueue.forEach(prom => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });

  isRefreshing = false;
  failedQueue = [];
};

client.interceptors.response.use(
  response => response,
  async error => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && originalRequest && !originalRequest._retry) {
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then(token => {
          originalRequest.headers['Authorization'] = `Bearer ${token}`;
          return client(originalRequest);
        });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      if (refreshToken) {
        try {
          const response = await axios.post(`${API_URL}/api/auth/refresh`, {
            refreshToken
          });

          const { accessToken: newAccessToken, refreshToken: newRefreshToken } = response.data;
          setTokens(newAccessToken, newRefreshToken);
          originalRequest.headers['Authorization'] = `Bearer ${newAccessToken}`;

          processQueue(null, newAccessToken);
          return client(originalRequest);
        } catch (err) {
          processQueue(err, null);
          clearTokens();
          window.location.href = '/login';
          return Promise.reject(err);
        }
      } else {
        clearTokens();
        window.location.href = '/login';
        return Promise.reject(error);
      }
    }

    return Promise.reject(error);
  }
);

export default client;
