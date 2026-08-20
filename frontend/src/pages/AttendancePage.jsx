import React, { useState, useEffect } from 'react';
import { attendanceApi } from '../api/attendance';
import { StatusBadge } from '../components/StatusBadge';
import { ConfirmModal } from '../components/ConfirmModal';

export function AttendancePage() {
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [checkoutTarget, setCheckoutTarget] = useState(null);
  const [actionLoading, setActionLoading] = useState(false);

  const loadAttendance = async (dateStr) => {
    setLoading(true);
    setError(null);
    try {
      const data = await attendanceApi.getByDate(dateStr);
      setRecords(data || []);
    } catch (err) {
      setError(err.message || 'Failed to load attendance records');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAttendance(selectedDate);
  }, [selectedDate]);

  const handleCheckoutConfirm = async () => {
    if (!checkoutTarget) return;
    setActionLoading(true);
    try {
      await attendanceApi.checkout(checkoutTarget.employee_id);
      setCheckoutTarget(null);
      await loadAttendance(selectedDate);
    } catch (err) {
      alert(`Checkout failed: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const handleExportCSV = () => {
    if (records.length === 0) return;
    const headers = ['ID', 'Employee Name', 'Employee Code', 'Date', 'Check-In', 'Check-Out', 'Confidence'];
    const rows = records.map((r) => [
      r.id,
      r.employee_name || `Employee #${r.employee_id}`,
      r.employee_code || '',
      r.date,
      new Date(r.check_in).toLocaleTimeString(),
      r.check_out ? new Date(r.check_out).toLocaleTimeString() : '',
      `${(r.confidence * 100).toFixed(1)}%`,
    ]);

    const csvContent = [headers.join(','), ...rows.map((row) => row.map((cell) => `"${cell}"`).join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `attendance_${selectedDate}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredRecords = records.filter((r) => {
    const q = searchQuery.toLowerCase();
    const name = (r.employee_name || '').toLowerCase();
    const code = (r.employee_code || '').toLowerCase();
    return name.includes(q) || code.includes(q);
  });

  const presentCount = records.length;
  const checkedOutCount = records.filter((r) => r.check_out !== null).length;

  return (
    <div className="page-container">
      {error && (
        <div className="banner banner-error" role="alert">
          <span className="banner-icon">⚠️</span>
          <span>{error}</span>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => loadAttendance(selectedDate)}>
            Retry
          </button>
        </div>
      )}

      {/* Date & Filter Control Bar */}
      <div className="controls-bar">
        <div className="controls-left">
          <div className="date-picker-group">
            <label className="form-label text-xs" htmlFor="att-date-picker">Filter by Date</label>
            <input
              id="att-date-picker"
              type="date"
              className="input-field date-input"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
            />
          </div>

          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => setSelectedDate(new Date().toISOString().split('T')[0])}
          >
            Today
          </button>

          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => {
              const d = new Date();
              d.setDate(d.getDate() - 1);
              setSelectedDate(d.toISOString().split('T')[0]);
            }}
          >
            Yesterday
          </button>
        </div>

        <div className="controls-right">
          <div className="search-input-wrapper">
            <input
              type="search"
              className="input-field"
              placeholder="Search by name or code..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleExportCSV}
            disabled={records.length === 0}
          >
            Export CSV
          </button>
        </div>
      </div>

      {/* Summary Stat Line */}
      <div className="attendance-summary-line">
        <span>Logged Date: <strong>{selectedDate}</strong></span>
        <span>Total Present: <strong>{presentCount}</strong></span>
        <span>Explicit Departures: <strong>{checkedOutCount}</strong></span>
      </div>

      {/* Records Table */}
      <div className="table-container">
        {loading ? (
          <div className="state-box">Loading attendance logs for {selectedDate}...</div>
        ) : filteredRecords.length === 0 ? (
          <div className="state-box state-empty">
            <p className="empty-title">No attendance records found for {selectedDate}.</p>
            <p className="empty-desc">
              {searchQuery ? 'No records matched your search query.' : 'No employees checked in on this date.'}
            </p>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Employee</th>
                <th>Employee Code</th>
                <th>Check-In Time</th>
                <th>Check-Out Time</th>
                <th>Verification Confidence</th>
                <th>Status</th>
                <th className="th-actions">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredRecords.map((rec) => {
                const isCheckedOut = rec.check_out !== null;
                const checkInTime = new Date(rec.check_in).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                const checkOutTime = rec.check_out ? new Date(rec.check_out).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—';
                const confPercent = `${(rec.confidence * 100).toFixed(1)}%`;

                return (
                  <tr key={rec.id}>
                    <td className="td-mono text-muted">#{rec.id}</td>
                    <td className="td-strong">{rec.employee_name || `Employee #${rec.employee_id}`}</td>
                    <td><code className="code-badge">{rec.employee_code || '—'}</code></td>
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
                          onClick={() => setCheckoutTarget(rec)}
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
        title="Confirm Explicit Checkout"
        message={`Confirm departure for ${checkoutTarget?.employee_name || `Employee #${checkoutTarget?.employee_id}`} on ${selectedDate}?`}
        confirmText={actionLoading ? 'Processing...' : 'Confirm Checkout'}
        onConfirm={handleCheckoutConfirm}
        onCancel={() => setCheckoutTarget(null)}
      />
    </div>
  );
}
