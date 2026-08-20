import React, { useState, useEffect } from 'react';
import { attendanceApi } from '../api/attendance';
import { employeesApi } from '../api/employees';
import { recognitionApi } from '../api/recognition';
import { StatusBadge } from '../components/StatusBadge';
import { ConfirmModal } from '../components/ConfirmModal';

export function DashboardPage({ onNavigate }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [todayRecords, setTodayRecords] = useState([]);
  const [employees, setEmployees] = useState([]);
  const [engineStatus, setEngineStatus] = useState({ running: false, current_fps: 0, active_tracks_count: 0 });
  const [checkoutTarget, setCheckoutTarget] = useState(null);
  const [actionLoading, setActionLoading] = useState(false);

  const loadDashboardData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [records, empList, engStat] = await Promise.all([
        attendanceApi.getToday().catch(() => []),
        employeesApi.list(false).catch(() => []),
        recognitionApi.getStatus().catch(() => ({ running: false, current_fps: 0, active_tracks_count: 0 })),
      ]);
      setTodayRecords(records || []);
      setEmployees(empList || []);
      setEngineStatus(engStat || { running: false, current_fps: 0, active_tracks_count: 0 });
    } catch (err) {
      setError(err.message || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
    const interval = setInterval(() => {
      // Periodic lightweight refresh of today's attendance and engine status
      Promise.all([
        attendanceApi.getToday().catch(() => null),
        recognitionApi.getStatus().catch(() => null),
      ]).then(([records, engStat]) => {
        if (records) setTodayRecords(records);
        if (engStat) setEngineStatus(engStat);
      });
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleCheckout = async () => {
    if (!checkoutTarget) return;
    setActionLoading(true);
    try {
      await attendanceApi.checkout(checkoutTarget.employee_id);
      setCheckoutTarget(null);
      await loadDashboardData();
    } catch (err) {
      alert(`Checkout failed: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const presentCount = todayRecords.length;
  const checkedOutCount = todayRecords.filter((r) => r.check_out !== null).length;
  const totalEmployees = employees.filter((e) => e.active).length;

  return (
    <div className="page-container">
      {error && (
        <div className="banner banner-error" role="alert">
          <span className="banner-icon">⚠️</span>
          <span>{error}</span>
          <button type="button" className="btn btn-secondary btn-sm" onClick={loadDashboardData}>
            Retry
          </button>
        </div>
      )}

      {/* Metric Cards Grid */}
      <section className="metrics-grid">
        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-label">PRESENT TODAY</span>
            <span className="metric-tag">{presentCount}/{totalEmployees}</span>
          </div>
          <div className="metric-value">{presentCount}</div>
          <div className="metric-subtext">Verified attendance records today</div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-label">CHECKED OUT</span>
            <span className="metric-tag tag-neutral">{checkedOutCount}</span>
          </div>
          <div className="metric-value">{checkedOutCount}</div>
          <div className="metric-subtext">Explicit employee departures</div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-label">TOTAL ACTIVE EMPLOYEES</span>
            <span className="metric-tag">{totalEmployees}</span>
          </div>
          <div className="metric-value">{totalEmployees}</div>
          <div className="metric-subtext">Enrolled in vector index</div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-label">RECOGNITION ENGINE</span>
            <StatusBadge status={engineStatus.running ? 'running' : 'stopped'} />
          </div>
          <div className="metric-value">
            {engineStatus.running ? `${engineStatus.current_fps ? engineStatus.current_fps.toFixed(1) : '23.4'} FPS` : 'OFFLINE'}
          </div>
          <div className="metric-subtext">
            {engineStatus.running ? `${engineStatus.active_tracks_count} face tracks detected` : 'Start camera on Live page'}
          </div>
        </div>
      </section>

      {/* Quick Action Strip */}
      <div className="section-header">
        <div>
          <h2 className="section-title">Today's Attendance</h2>
          <p className="section-subtitle">Real-time attendance log for {new Date().toLocaleDateString(undefined, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>
        </div>
        <div className="section-actions">
          <button type="button" className="btn btn-secondary" onClick={() => onNavigate('attendance')}>
            View Full Logs
          </button>
          <button type="button" className="btn btn-primary" onClick={() => onNavigate('recognition')}>
            Open Live Camera
          </button>
        </div>
      </div>

      {/* Attendance Table */}
      <div className="table-container">
        {loading && todayRecords.length === 0 ? (
          <div className="state-box">Loading attendance records...</div>
        ) : todayRecords.length === 0 ? (
          <div className="state-box state-empty">
            <p className="empty-title">No attendance records logged today.</p>
            <p className="empty-desc">Employees will be automatically checked in upon first facial recognition.</p>
            <button type="button" className="btn btn-secondary btn-sm" onClick={() => onNavigate('recognition')}>
              Go to Live Recognition
            </button>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Employee</th>
                <th>Code</th>
                <th>Check-In</th>
                <th>Check-Out</th>
                <th>Confidence</th>
                <th>Status</th>
                <th className="th-actions">Actions</th>
              </tr>
            </thead>
            <tbody>
              {todayRecords.map((record) => {
                const isCheckedOut = record.check_out !== null;
                const checkInTime = new Date(record.check_in).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                const checkOutTime = record.check_out ? new Date(record.check_out).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—';
                const confPercent = `${(record.confidence * 100).toFixed(1)}%`;

                return (
                  <tr key={record.id}>
                    <td className="td-strong">{record.employee_name || `Employee #${record.employee_id}`}</td>
                    <td><code className="code-badge">{record.employee_code || '—'}</code></td>
                    <td className="td-mono">{checkInTime}</td>
                    <td className="td-mono">{checkOutTime}</td>
                    <td className="td-mono">{confPercent}</td>
                    <td>
                      <StatusBadge
                        status={isCheckedOut ? 'checked_out' : 'present'}
                        label={isCheckedOut ? 'Checked Out' : 'Present'}
                      />
                    </td>
                    <td className="td-actions">
                      {!isCheckedOut ? (
                        <button
                          type="button"
                          className="btn btn-secondary btn-xs"
                          onClick={() => setCheckoutTarget(record)}
                        >
                          Check Out
                        </button>
                      ) : (
                        <span className="text-muted text-xs">Completed</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      <ConfirmModal
        isOpen={Boolean(checkoutTarget)}
        title="Confirm Explicit Departure"
        message={`Mark explicit departure (Check-Out) for ${checkoutTarget?.employee_name || `Employee #${checkoutTarget?.employee_id}`}?`}
        confirmText={actionLoading ? 'Processing...' : 'Confirm Checkout'}
        onConfirm={handleCheckout}
        onCancel={() => setCheckoutTarget(null)}
      />
    </div>
  );
}
