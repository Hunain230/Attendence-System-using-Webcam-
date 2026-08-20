import { api } from './client';

export const recognitionApi = {
  start: () => api.post('/api/recognition/start'),
  stop: () => api.post('/api/recognition/stop'),
  getStatus: () => api.get('/api/recognition/status'),
  getStreamUrl: () => `${api.getBaseUrl()}/api/stream`,
};
