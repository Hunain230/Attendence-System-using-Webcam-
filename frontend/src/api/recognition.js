import { api } from './client';

export const recognitionApi = {
  start:           ()        => api.post('/api/recognition/start'),
  stop:            ()        => api.post('/api/recognition/stop'),
  getStatus:       ()        => api.get('/api/recognition/status'),
  getMetrics:      ()        => api.get('/api/recognition/metrics'),
  setDebugOverlay: (enabled) => api.post(`/api/recognition/debug/${enabled}`),
  getStreamUrl:    ()        => `${api.getBaseUrl()}/api/stream`,
};
