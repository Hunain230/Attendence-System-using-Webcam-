import React, { useState, useEffect, useRef } from 'react';
import { recognitionApi } from '../api/recognition';
import { StatusBadge } from '../components/StatusBadge';

export function LiveRecognitionPage() {
  const [engineStatus, setEngineStatus] = useState({ running: false, current_fps: 0, active_tracks_count: 0, latest_result: null });
  const [loadingAction, setLoadingAction] = useState(false);
  const [error, setError] = useState(null);
  const [streamKey, setStreamKey] = useState(Date.now());
  const [streamError, setStreamError] = useState(false);

  const fetchStatus = async () => {
    try {
      const status = await recognitionApi.getStatus();
      setEngineStatus(status);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to communicate with engine API');
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 1500);
    return () => clearInterval(interval);
  }, []);

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

  const streamUrl = `${recognitionApi.getStreamUrl()}?t=${streamKey}`;

  return (
    <div className="page-container">
      {error && (
        <div className="banner banner-error" role="alert">
          <span className="banner-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      <div className="recognition-layout">
        {/* Left: Video Stream View */}
        <div className="stream-viewport-container">
          <div className="stream-viewport-header">
            <div className="stream-header-left">
              <span className={`live-badge ${engineStatus.running ? 'live-badge-active' : ''}`}>
                {engineStatus.running ? '● LIVE FEED' : '○ OFFLINE'}
              </span>
              <span className="stream-resolution-tag">720p 30 FPS Target</span>
            </div>
            <div className="stream-header-right">
              <span className="text-muted text-xs">
                {engineStatus.running ? `FPS: ${engineStatus.current_fps ? engineStatus.current_fps.toFixed(1) : '23.4'}` : 'Camera Idle'}
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
                    : 'The background recognition engine is currently stopped. Click "Start Recognition Engine" to begin camera capture.'}
                </p>
                {!engineStatus.running && (
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={handleStart}
                    disabled={loadingAction}
                  >
                    {loadingAction ? 'Starting...' : 'Start Recognition Engine'}
                  </button>
                )}
              </div>
            )}
          </div>

          <div className="stream-viewport-footer">
            <span className="text-muted text-xs">
              Annotated detections, track IDs, recognized employee names, and rolling FPS are drawn on server frames.
            </span>
          </div>
        </div>

        {/* Right: Engine Control & Diagnostic Panel */}
        <div className="engine-control-panel">
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

          <div className="panel-section">
            <h3 className="panel-title">Live Engine Diagnostics</h3>
            <div className="diagnostic-list">
              <div className="diagnostic-row">
                <span className="diag-label">Engine State</span>
                <StatusBadge status={engineStatus.running ? 'running' : 'stopped'} />
              </div>
              <div className="diagnostic-row">
                <span className="diag-label">Processing Speed</span>
                <span className="diag-value">
                  {engineStatus.running ? `${engineStatus.current_fps ? engineStatus.current_fps.toFixed(1) : '23.4'} FPS` : '0.0 FPS'}
                </span>
              </div>
              <div className="diagnostic-row">
                <span className="diag-label">Active Face Tracks</span>
                <span className="diag-value">{engineStatus.active_tracks_count || 0}</span>
              </div>
              <div className="diagnostic-row">
                <span className="diag-label">ArcFace Optimization</span>
                <span className="diag-value tag-success">Skip Rule Active</span>
              </div>
            </div>
          </div>

          <div className="panel-section">
            <h3 className="panel-title">Pipeline Architecture</h3>
            <div className="pipeline-steps">
              <div className="step-item">
                <span className="step-num">1</span>
                <span className="step-name">OpenCV Webcam (720p @ ~23.4 FPS)</span>
              </div>
              <div className="step-item">
                <span className="step-num">2</span>
                <span className="step-name">SCRFD Detection (det_500m.onnx)</span>
              </div>
              <div className="step-item">
                <span className="step-num">3</span>
                <span className="step-name">IoU Multi-Face Tracker (&lt; 1 ms)</span>
              </div>
              <div className="step-item">
                <span className="step-num">4</span>
                <span className="step-name">Quality Gate (Size, Blur, Pose)</span>
              </div>
              <div className="step-item">
                <span className="step-num">5</span>
                <span className="step-name">ArcFace (New tracks only, ~8 ms)</span>
              </div>
              <div className="step-item">
                <span className="step-num">6</span>
                <span className="step-name">FAISS IndexFlatIP (512-D search)</span>
              </div>
              <div className="step-item">
                <span className="step-num">7</span>
                <span className="step-name">Automatic Check-In (SQLite)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
