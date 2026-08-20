import React, { useState, useEffect } from 'react';
import { employeesApi } from '../api/employees';
import { StatusBadge } from '../components/StatusBadge';
import { ConfirmModal } from '../components/ConfirmModal';

export function EmployeesPage({ onNavigateToEnrollment }) {
  const [employees, setEmployees] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeOnly, setActiveOnly] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [deactivateTarget, setDeactivateTarget] = useState(null);
  const [formLoading, setFormLoading] = useState(false);
  const [formError, setFormError] = useState(null);

  // Form State
  const [formData, setFormData] = useState({
    employee_code: '',
    name: '',
    department: '',
  });

  const loadEmployees = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await employeesApi.list(activeOnly);
      setEmployees(data || []);
    } catch (err) {
      setError(err.message || 'Failed to load employee list');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadEmployees();
  }, [activeOnly]);

  const handleCreateSubmit = async (e) => {
    e.preventDefault();
    if (!formData.employee_code.trim() || !formData.name.trim()) {
      setFormError('Employee Code and Full Name are required.');
      return;
    }

    setFormLoading(true);
    setFormError(null);
    try {
      await employeesApi.create({
        employee_code: formData.employee_code.trim(),
        name: formData.name.trim(),
        department: formData.department.trim() || null,
      });
      setShowCreateModal(false);
      setFormData({ employee_code: '', name: '', department: '' });
      await loadEmployees();
    } catch (err) {
      setFormError(err.message || 'Failed to create employee');
    } finally {
      setFormLoading(false);
    }
  };

  const handleDeactivateConfirm = async () => {
    if (!deactivateTarget) return;
    try {
      await employeesApi.deactivate(deactivateTarget.id);
      setDeactivateTarget(null);
      await loadEmployees();
    } catch (err) {
      alert(`Deactivation failed: ${err.message}`);
    }
  };

  const filteredEmployees = employees.filter((emp) => {
    const q = searchQuery.toLowerCase();
    const matchesQuery =
      emp.name.toLowerCase().includes(q) ||
      emp.employee_code.toLowerCase().includes(q) ||
      (emp.department && emp.department.toLowerCase().includes(q));
    return matchesQuery;
  });

  return (
    <div className="page-container">
      {error && (
        <div className="banner banner-error" role="alert">
          <span className="banner-icon">⚠️</span>
          <span>{error}</span>
          <button type="button" className="btn btn-secondary btn-sm" onClick={loadEmployees}>
            Retry
          </button>
        </div>
      )}

      {/* Control Strip */}
      <div className="controls-bar">
        <div className="search-input-wrapper">
          <input
            type="search"
            className="input-field"
            placeholder="Search by name, code, or department..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div className="controls-right">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={activeOnly}
              onChange={(e) => setActiveOnly(e.target.checked)}
            />
            <span>Active only</span>
          </label>

          <button
            type="button"
            className="btn btn-primary"
            onClick={() => {
              setFormError(null);
              setShowCreateModal(true);
            }}
          >
            + Add Employee
          </button>
        </div>
      </div>

      {/* Employees Table */}
      <div className="table-container">
        {loading && employees.length === 0 ? (
          <div className="state-box">Loading employees...</div>
        ) : filteredEmployees.length === 0 ? (
          <div className="state-box state-empty">
            <p className="empty-title">No employees found.</p>
            <p className="empty-desc">
              {searchQuery ? 'No employees matched your search criteria.' : 'No registered employees in database yet.'}
            </p>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Employee Code</th>
                <th>Full Name</th>
                <th>Department</th>
                <th>Registered Date</th>
                <th>Status</th>
                <th className="th-actions">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredEmployees.map((emp) => (
                <tr key={emp.id} className={!emp.active ? 'tr-muted' : ''}>
                  <td className="td-mono text-muted">#{emp.id}</td>
                  <td><code className="code-badge">{emp.employee_code}</code></td>
                  <td className="td-strong">{emp.name}</td>
                  <td>{emp.department || '—'}</td>
                  <td className="td-mono">{new Date(emp.created_at).toLocaleDateString()}</td>
                  <td>
                    <StatusBadge
                      status={emp.active ? 'active' : 'inactive'}
                      label={emp.active ? 'Active' : 'Deactivated'}
                    />
                  </td>
                  <td className="td-actions">
                    {emp.active ? (
                      <>
                        <button
                          type="button"
                          className="btn btn-secondary btn-xs"
                          onClick={() => onNavigateToEnrollment(emp)}
                          title="Enroll facial samples for this employee"
                        >
                          Enroll Faces
                        </button>
                        <button
                          type="button"
                          className="btn btn-danger-outline btn-xs"
                          onClick={() => setDeactivateTarget(emp)}
                        >
                          Deactivate
                        </button>
                      </>
                    ) : (
                      <span className="text-muted text-xs">Deactivated</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Create Employee Modal */}
      {showCreateModal && (
        <div className="modal-backdrop" onClick={() => setShowCreateModal(false)} role="dialog" aria-modal="true">
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">Register New Employee</h3>
            </div>
            <form onSubmit={handleCreateSubmit}>
              <div className="modal-body form-grid">
                {formError && <div className="form-error-banner">{formError}</div>}

                <div className="form-group">
                  <label className="form-label" htmlFor="emp-code">
                    Employee Code <span className="req">*</span>
                  </label>
                  <input
                    id="emp-code"
                    type="text"
                    className="input-field"
                    placeholder="e.g. EMP-1042"
                    required
                    value={formData.employee_code}
                    onChange={(e) => setFormData({ ...formData, employee_code: e.target.value })}
                  />
                  <span className="form-hint">Unique identifier for vector ID mapping</span>
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="emp-name">
                    Full Name <span className="req">*</span>
                  </label>
                  <input
                    id="emp-name"
                    type="text"
                    className="input-field"
                    placeholder="e.g. Hunain Shahid"
                    required
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="emp-dept">
                    Department
                  </label>
                  <input
                    id="emp-dept"
                    type="text"
                    className="input-field"
                    placeholder="e.g. Artificial Intelligence"
                    value={formData.department}
                    onChange={(e) => setFormData({ ...formData, department: e.target.value })}
                  />
                </div>
              </div>

              <div className="modal-actions">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setShowCreateModal(false)}
                  disabled={formLoading}
                >
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary" disabled={formLoading}>
                  {formLoading ? 'Creating...' : 'Create Employee'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Deactivate Confirm Modal */}
      <ConfirmModal
        isOpen={Boolean(deactivateTarget)}
        title="Deactivate Employee Record"
        message={`Are you sure you want to deactivate ${deactivateTarget?.name} (${deactivateTarget?.employee_code})? This will soft-delete their database record and remove their face vectors from the FAISS matcher.`}
        confirmText="Deactivate"
        isDanger={true}
        onConfirm={handleDeactivateConfirm}
        onCancel={() => setDeactivateTarget(null)}
      />
    </div>
  );
}
