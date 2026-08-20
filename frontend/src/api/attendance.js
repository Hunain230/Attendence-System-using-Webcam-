import { api } from './client';

export const attendanceApi = {
  getToday: () => api.get('/api/attendance/today'),
  getByDate: (dateStr) => api.get(`/api/attendance?date=${dateStr}`),
  getByEmployee: (employeeId) => api.get(`/api/attendance/employee/${employeeId}`),
  checkout: (employeeId) => api.post(`/api/attendance/${employeeId}/checkout`),
};
