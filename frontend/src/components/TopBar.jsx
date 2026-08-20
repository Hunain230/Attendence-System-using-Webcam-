import React, { useState, useEffect } from 'react';
import { StatusBadge } from './StatusBadge';

export function TopBar({ title, subtitle, backendOnline, engineRunning, currentFps, onRefresh, isRefreshing }) {
  const [timeStr, setTimeStr] = useState('');

  useEffect(() => {
    const update = () => {
      const now = new Date();
      setTimeStr(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    };
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="app-topbar">
      <div className="topbar-left">
        <h1 className="topbar-title">{title}</h1>
        {subtitle && <span className="topbar-subtitle">{subtitle}</span>}
      </div>

      <div className="topbar-right">
        <div className="topbar-meta">
          <span className="meta-time">{timeStr}</span>
          <StatusBadge
            status={engineRunning ? 'running' : 'stopped'}
            label={engineRunning ? `Engine: ${currentFps ? `${currentFps.toFixed(1)} FPS` : 'Running'}` : 'Engine: Stopped'}
          />
          <StatusBadge
            status={backendOnline ? 'online' : 'offline'}
            label={backendOnline ? 'FastAPI 1.0.0' : 'Backend Offline'}
          />
        </div>

        {onRefresh && (
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onRefresh}
            disabled={isRefreshing}
            title="Refresh current page data"
          >
            {isRefreshing ? 'Refreshing...' : 'Refresh'}
          </button>
        )}
      </div>
    </header>
  );
}
