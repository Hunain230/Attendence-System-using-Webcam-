import React from 'react';

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', icon: '📊' },
  { id: 'recognition', label: 'Live Recognition', icon: '📷' },
  { id: 'employees', label: 'Employees', icon: '👥' },
  { id: 'enrollment', label: 'Enrollment', icon: '👤' },
  { id: 'attendance', label: 'Attendance Logs', icon: '📋' },
];

export function Sidebar({ activeTab, onTabChange, backendOnline, engineRunning }) {
  return (
    <aside className="app-sidebar" aria-label="Sidebar navigation">
      <div className="sidebar-brand">
        <div className="brand-logo">FACERECOG</div>
        <div className="brand-title">ATTENDANCE</div>
        <div className="brand-subtitle">Offline-First System</div>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              type="button"
              className={`nav-item ${isActive ? 'nav-item-active' : ''}`}
              onClick={() => onTabChange(item.id)}
              aria-current={isActive ? 'page' : undefined}
            >
              <span className="nav-icon" aria-hidden="true">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
              {item.id === 'recognition' && engineRunning && (
                <span className="nav-tag tag-live">LIVE</span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="system-indicator">
          <span className={`indicator-dot ${backendOnline ? 'dot-online' : 'dot-offline'}`} />
          <span className="indicator-text">
            {backendOnline ? 'API: Connected' : 'API: Offline'}
          </span>
        </div>
        <div className="system-indicator">
          <span className={`indicator-dot ${engineRunning ? 'dot-online' : 'dot-idle'}`} />
          <span className="indicator-text">
            {engineRunning ? 'Engine: Active' : 'Engine: Idle'}
          </span>
        </div>
      </div>
    </aside>
  );
}
