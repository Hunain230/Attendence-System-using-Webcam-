import React, { useState, useEffect, useRef } from 'react';
import { recognitionApi } from '../api/recognition';
import { StatusBadge } from '../components/StatusBadge';

export function LiveRecognitionPage() {
  const [engineStatus, setEngineStatus] = useState({
    running: false, current_fps: 0, active_tracks_count: 0, latest_result: null,
  });
  const [metrics, setMetrics] = useState(null);
  const [loadingAction, setLoadingAction] = useState(false);
  const [error, setError] = useState(null);
  const [streamKey, setStreamKey] = useState(Date.now());
  const [streamError, setStreamError] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [overlayEnabled, setOverlayEnabled] = useState(false);
  const metricsIntervalRef = useRef(null);

  const fetchStatus = async () => {
    try {
      const status = await recognitionApi.getStatus();
      setEngineStatus(status);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to communicate with engine API');
    }
  };

  const fetchMetrics = async () => {
    try {
      const m = await recognitionApi.getMetrics();
      setMetrics(m);
    } catch {}
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 1500);
    return () => clearInterval(interval);
  }, []);

  // Poll metrics only when debug panel is open
  useEffect(() => {
    if (showDebug) {
      fetchMetrics();
      metricsIntervalRef.current = setInterval(fetchMetrics, 2000);
    } else {
      if (metricsIntervalRef.current) clearInterval(metricsIntervalRef.current);
    }
    return () => {
      if (metricsIntervalRef.current) clearInterval(metricsIntervalRef.current);
    };
  }, [showDebug]);

  const handleStart = async () => {
    setLoadingAction(true);
    setError(null);
    try {
      await recognitionApi.start();
      setStreamError(false);
      setStreamKey(Date.now());
      await fetchStatus();
    } catch (err) {
      setError(err.message || 'Failed to start engine');
    } finally {
      setLoadingAction(false);
    }
  };

  const handleStop = async () => {
    setLoadingAction(true);
    setError(null);
    try {
      await recognitionApi.stop();
      await fetchStatus();
    } catch (err) {
      setError(err.message || 'Failed to stop engine');
    } finally {
      setLoadingAction(false);
    }
  };

  const handleToggleOverlay = async () => {
    const next = !overlayEnabled;
    try {
      await recognitionApi.setDebugOverlay(next);
      setOverlayEnabled(next);
    } catch {}
  };

  const streamUrl = `${recognitionApi.getStreamUrl()}?t=${streamKey}`;

  const MetricRow = ({ label, value, unit = '', highlight = false }) => (
    <div className="diagnostic-row">
      <span className="diag-label">{label}</span>
      <span className={`diag-value ${highlight ? 'tag-success' : ''}`}>
        {value != null ? `${value}${unit}` : '—'}
      </span>
    </div>
  );

  return (
    <div className="page-container">
      {error && (
        <div className="banner banner-error" role="alert">
          <span className="banner-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      <div className="recognition-layout">
        {/* Left: Video Stream */}
        <div className="stream-viewport-container">
          <div className="stream-viewport-header">
            <div className="stream-header-left">
              <span className={`live-badge ${engineStatus.running ? 'live-badge-active' : ''}`}>
                {engineStatus.running ? '● LIVE FEED' : '○ OFFLINE'}
              </span>
              <span className="stream-resolution-tag">720p | Adaptive Recognition</span>
            </div>
            <div className="stream-header-right">
              <span className="text-muted text-xs">
                {engineStatus.running
                  ? `FPS: ${metrics?.fps != null ? metrics.fps.toFixed(1) : engineStatus.current_fps?.toFixed(1) ?? '—'}`
                  : 'Camera Idle'}
              </span>
            </div>
          </div>

          <div className="stream-viewport">
            {engineStatus.running && !streamError ? (
              <img
                key={streamKey}
                src={streamUrl}
                alt="Live Annotated Camera Feed"
                className="stream-image"
                onError={() => setStreamError(true)}
              />
            ) : (
              <div className="stream-offline-placeholder">
                <div className="placeholder-icon">📷</div>
                <div className="placeholder-title">Camera Stream Inactive</div>
                <p className="placeholder-desc">
                  {streamError
                    ? 'Failed to connect to video stream endpoint. Start engine below.'
                    : 'The background recognition engine is stopped. Click "Start Engine" to begin.'}
                </p>
                {!engineStatus.running && (
                  <button type="button" className="btn btn-primary" onClick={handleStart} disabled={loadingAction}>
                    {loadingAction ? 'Starting...' : 'Start Recognition Engine'}
                  </button>
                )}
              </div>
            )}
          </div>

          <div className="stream-viewport-footer">
            <span className="text-muted text-xs">
              Green = Confirmed | Orange = Liveness Warn | Cyan = Checking | Red = Unknown
              {overlayEnabled ? ' | Debug overlay ON' : ''}
            </span>
          </div>
        </div>

        {/* Right: Control + Diagnostics */}
        <div className="engine-control-panel">
          {/* Engine Controls */}
          <div className="panel-section">
            <h3 className="panel-title">Engine Controls</h3>
            <div className="control-button-group">
              <button
                type="button"
                className="btn btn-primary btn-block"
                onClick={handleStart}
                disabled={engineStatus.running || loadingAction}
              >
                {loadingAction && !engineStatus.running ? 'Starting...' : 'Start Engine'}
              </button>
              <button
                type="button"
                className="btn btn-secondary btn-block"
                onClick={handleStop}
                disabled={!engineStatus.running || loadingAction}
              >
                {loadingAction && engineStatus.running ? 'Stopping...' : 'Stop Engine'}
              </button>
            </div>
          </div>

          {/* Status */}
          <div className="panel-section">
            <h3 className="panel-title">Engine Status</h3>
            <div className="diagnostic-list">
              <MetricRow label="Engine State" value={engineStatus.running ? 'Running' : 'Stopped'} highlight={engineStatus.running} />
              <MetricRow label="Active Face Tracks" value={engineStatus.active_tracks_count ?? 0} />
              <MetricRow label="Recognition Strategy" value="Adaptive (state-based)" />
              <MetricRow label="Liveness Protection" value="Active" highlight />
            </div>
          </div>

          {/* Debug Toggle */}
          <div className="panel-section">
            <h3 className="panel-title">Diagnostics</h3>
            <div className="control-button-group" style={{ flexDirection: 'column', gap: 8 }}>
              <label className="toggle-switch-label">
                <input type="checkbox" checked={showDebug} onChange={(e) => setShowDebug(e.target.checked)} />
                <span className="toggle-text"><strong>Show Performance Metrics</strong></span>
              </label>
              <label className="toggle-switch-label">
                <input
                  type="checkbox"
                  checked={overlayEnabled}
                  onChange={handleToggleOverlay}
                  disabled={!engineStatus.running}
                />
                <span className="toggle-text"><strong>Debug Overlay on Stream</strong> (shows yaw/pitch/similarity)</span>
              </label>
            </div>
          </div>

          {/* Metrics Panel */}
          {showDebug && (
            <div className="panel-section">
              <h3 className="panel-title">Performance Metrics</h3>
              <div className="diagnostic-list">
                <MetricRow label="Capture FPS" value={metrics?.fps?.toFixed(1)} />
                <MetricRow label="Detection Latency" value={metrics?.detection_latency_ms?.toFixed(1)} unit=" ms" />
                <MetricRow label="ArcFace Latency" value={metrics?.arcface_latency_ms?.toFixed(1)} unit=" ms" />
                <MetricRow label="Total Latency" value={metrics?.total_latency_ms?.toFixed(1)} unit=" ms" />
                <MetricRow label="ArcFace Calls" value={metrics?.arcface_invocations} />
                <MetricRow label="ArcFace Skipped" value={metrics?.arcface_skipped} />
                <MetricRow label="CPU Usage" value={metrics?.cpu_percent?.toFixed(1)} unit="%" />
                <MetricRow label="RAM Usage" value={metrics?.ram_mb?.toFixed(0)} unit=" MB" />
              </div>
              <p style={{ color: '#64748b', fontSize: '0.78rem', marginTop: 8 }}>
                Updated every 2s. Run <code>python backend/calibrate.py</code> to get threshold recommendations.
              </p>
            </div>
          )}

          {/* Pipeline Summary */}
          <div className="panel-section">
            <h3 className="panel-title">Pipeline Architecture</h3>
            <div className="pipeline-steps">
              {[
                'Camera Capture Thread (full speed)',
                'Frame Queue (drop-old, max 2)',
                'Recognition Thread (SCRFD every 2 frames)',
                'IoU Face Tracker (every frame)',
                'Liveness Check (optical flow + LBP)',
                'Quality Gate (size, blur, pose)',
                'ArcFace Embedding (adaptive schedule)',
                'FAISS + Margin Check (unconditional)',
                'Temporal Voting (5-frame majority)',
                'Attendance Check-In (confirmed only)',
              ].map((step, i) => (
                <div className="step-item" key={i}>
                  <span className="step-num">{i + 1}</span>
                  <span className="step-name">{step}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
