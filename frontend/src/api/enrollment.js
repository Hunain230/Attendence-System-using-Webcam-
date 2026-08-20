import { api } from './client';

export const enrollmentApi = {
  start: (data) => api.post('/api/enrollment/start', data),
  sendSample: (payload) => api.post('/api/enrollment/sample', payload),
};
