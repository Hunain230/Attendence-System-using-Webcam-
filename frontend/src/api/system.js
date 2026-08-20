import { api } from './client';

export const systemApi = {
  health: () => api.get('/health'),
};
