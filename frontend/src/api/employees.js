import { api } from './client';

export const employeesApi = {
  list: (activeOnly = true) => api.get(`/api/employees?active_only=${activeOnly}`),
  getById: (id) => api.get(`/api/employees/${id}`),
  create: (data) => api.post('/api/employees', data),
  deactivate: (id) => api.delete(`/api/employees/${id}`),
};
