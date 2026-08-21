import { api } from './client';

export const enrollmentApi = {
  start:           (data)       => api.post('/api/enrollment/start', data),
  startSession:    (employeeId) => api.post(`/api/enrollment/start_session/${employeeId}`),
  sendSample:      (payload)    => api.post('/api/enrollment/sample', payload),
  resetEmbeddings: (employeeId) => api.post(`/api/enrollment/reset/${employeeId}`),
};
