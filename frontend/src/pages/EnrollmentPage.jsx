import React, { useState, useEffect, useRef, useCallback } from 'react';
import { enrollmentApi } from '../api/enrollment';
import { employeesApi } from '../api/employees';
import { StatusBadge } from '../components/StatusBadge';

const TOTAL_SAMPLES_TARGET = 7;

const POSE_DEFINITIONS = [
  { id: 'straight',     name: 'Straight (Front)',   icon: '🎯', desc: 'Look directly into the camera lens with neutral expression.' },
  { id: 'straight',     name: 'Straight (Angle 2)', icon: '🎯', desc: 'Maintain direct eye contact with the camera.' },
  { id: 'slight_left',  name: 'Turn Left',          icon: '⬅️', desc: 'Turn head slightly to the left (~20°). Eyes level.' },
  { id: 'slight_right', name: 'Turn Right',         icon: '➡️', desc: 'Turn head slightly to the right (~20°). Eyes level.' },
  { id: 'slight_up',    name: 'Tilt Up',            icon: '⬆️', desc: 'Tilt your chin slightly upward (~12°). Face camera.' },
  { id: 'slight_down',  name: 'Tilt Down',          icon: '⬇️', desc: 'Tilt your chin slightly downward (~12°). Face camera.' },
  { id: 'smile',        name: 'Smile Naturally',    icon: '😊', desc: 'Look straight and smile naturally.' },
];

// Phase → user-facing label and style
const PHASE_DISPLAY = {
  guidance:   { label: 'Adjusting...',        color: '#94a3b8' },
  holding:    { label: 'Hold still...',       color: '#f59e0b' },
  collecting: { label: 'Capturing best frame...', color: '#6366f1' },
  captured:   { label: '✓ Captured',          color: '#22c55e' },
  complete:   { label: '✓ Complete',          color: '#22c55e' },
};

export function EnrollmentPage({ preselectedEmployee }) {
  const [employees, setEmployees] = useState([]);
  const [selectedEmployee, setSelectedEmployee] = useState(preselectedEmployee || null);
  const [session, setSession] = useState(null);
  const [samplesCount, setSamplesCount] = useState(0);
  const [currentPose, setCurrentPose] = useState('straight');
  const [instructions, setInstructions] = useState(POSE_DEFINITIONS[0].desc);
  const [feedback, setFeedback] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [smartAutoSnap, setSmartAutoSnap] = useState(true);
  const [flashActive, setFlashActive] = useState(false);

  // Live metrics from backend
  const [liveMetrics, setLiveMetrics] = useState({
    yaw: 0, pitch: 0,
    guidance: 'Position face in camera frame',
    score: null, score_100: null,
    phase: 'guidance',
    hold_progress: 0, hold_required: 0,
    collect_progress: 0, collect_required: 0,
  });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const loopTimerRef = useRef(null);
  const isEvaluatingRef = useRef(false);

  // Sound chime on snapshot capture
  const playCaptureSound = () => {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1760, ctx.currentTime + 0.12);
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.12);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.13);
    } catch (_) {}
  };

  // Load employee list
  useEffect(() => {
    employeesApi.list(true).then((data) => {
      setEmployees(data || []);
      if (preselectedEmployee) {
        setSelectedEmployee(preselectedEmployee);
      } else if (data && data.length > 0 && !selectedEmployee) {
        setSelectedEmployee(data[0]);
      }
    });
  }, [preselectedEmployee]);

  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setCameraActive(true);
    } catch (err) {
      setCameraError(`Webcam access denied: ${err.message}. Please enable camera permissions.`);
      setCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  };

  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
      if (loopTimerRef.current) clearInterval(loopTimerRef.current);
    };
  }, []);

  const handleStartSession = async () => {
    if (!selectedEmployee) return;
    setFeedback(null);
    setSamplesCount(0);
    setLiveMetrics((m) => ({ ...m, phase: 'guidance', hold_progress: 0, collect_progress: 0 }));

    try {
      const startRes = await enrollmentApi.startSession(selectedEmployee.id);
      setSession(startRes);
      setCurrentPose(startRes.current_pose || 'straight');
      setInstructions(startRes.instructions || POSE_DEFINITIONS[0].desc);
      setFeedback({ type: 'info', message: 'Enrollment active. Follow the on-screen guidance.' });
    } catch {
      try {
        const startRes2 = await enrollmentApi.start({
          employee_code: selectedEmployee.employee_code,
          name: selectedEmployee.name,
          department: selectedEmployee.department || null,
        });
        setSession(startRes2);
        setCurrentPose(startRes2.current_pose || 'straight');
        setInstructions(startRes2.instructions || POSE_DEFINITIONS[0].desc);
      } catch {
        setSession({
          employee_id: selectedEmployee.id,
          employee_code: selectedEmployee.employee_code,
          name: selectedEmployee.name,
          status: 'in_progress',
          current_pose: 'straight',
          instructions: POSE_DEFINITIONS[0].desc,
        });
        setCurrentPose('straight');
        setInstructions(POSE_DEFINITIONS[0].desc);
      }
    }
  };

  const handleResetEnrollment = async () => {
    if (!selectedEmployee) return;
    setFeedback(null);
    setSamplesCount(0);
    setSession(null);
    setLiveMetrics((m) => ({ ...m, phase: 'guidance', hold_progress: 0, collect_progress: 0 }));

    try {
      const res = await enrollmentApi.resetEmbeddings(selectedEmployee.id);
      setSession(res);
      setCurrentPose(res.current_pose || 'straight');
      setInstructions(res.instructions || POSE_DEFINITIONS[0].desc);
      setFeedback({ type: 'info', message: 'Embeddings cleared. Starting fresh enrollment.' });
    } catch (err) {
      setFeedback({ type: 'error', message: `Reset failed: ${err.message}` });
    }
  };

  // Process a frame — does NOT claim success until backend returns captured=true
  const evaluateOrCaptureFrame = useCallback(async (forceCapture = false) => {
    if (!videoRef.current || !canvasRef.current || !session) return;
    if (samplesCount >= TOTAL_SAMPLES_TARGET) return;
    if (isEvaluatingRef.current) return;

    const video = videoRef.current;
    if (video.readyState < 2) return;

    isEvaluatingRef.current = true;
    setEvaluating(true);

    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.82);
    const base64Data = dataUrl.split(',')[1];

    try {
      const res = await enrollmentApi.sendSample({
        employee_id: session.employee_id,
        employee_code: session.employee_code,
        name: session.name,
        image_base64: base64Data,
        pose_name: currentPose,
        force_capture: forceCapture,
      });

      // Always update live metrics regardless of capture outcome
      setLiveMetrics({
        yaw: res.yaw ?? 0,
        pitch: res.pitch ?? 0,
        guidance: res.guidance || 'Position face',
        score: res.score ?? null,
        score_100: res.score_100 ?? null,
        phase: res.phase || 'guidance',
        hold_progress: res.hold_progress ?? 0,
        hold_required: res.hold_required ?? 0,
        collect_progress: res.collect_progress ?? 0,
        collect_required: res.collect_required ?? 0,
      });

      // ── Only trigger success UI when backend confirms captured=true ──
      if (res.captured) {
        // Backend committed the embedding — NOW show success
        playCaptureSound();
        setFlashActive(true);
        setTimeout(() => setFlashActive(false), 300);

        const newCount = res.samples_count || (samplesCount + 1);
        setSamplesCount(newCount);

        const nextPoseName = res.next_pose || 'complete';
        setCurrentPose(nextPoseName);

        const poseDef = POSE_DEFINITIONS[Math.min(newCount, POSE_DEFINITIONS.length - 1)];
        setInstructions(res.next_instructions || poseDef.desc);

        const qualityDisplay = res.score_100 != null
          ? `Quality: ${res.score_100.toFixed(1)}%`
          : res.score != null
            ? `Quality: ${(res.score * 100).toFixed(1)}%`
            : '';

        setFeedback({
          type: 'success',
          message: `📸 Captured! ${qualityDisplay} (${newCount}/${TOTAL_SAMPLES_TARGET})`,
        });
      }
      // If not captured, the liveMetrics guidance text explains why (no extra toast)
    } catch (err) {
      if (forceCapture) {
        setFeedback({ type: 'error', message: err.message || 'Frame processing error' });
      }
    } finally {
      isEvaluatingRef.current = false;
      setEvaluating(false);
    }
  }, [session, samplesCount, currentPose]);

  // Real-time evaluation loop (250ms polling)
  useEffect(() => {
    if (smartAutoSnap && session && samplesCount < TOTAL_SAMPLES_TARGET && cameraActive) {
      loopTimerRef.current = setInterval(() => evaluateOrCaptureFrame(false), 250);
    } else {
      if (loopTimerRef.current) clearInterval(loopTimerRef.current);
    }
    return () => { if (loopTimerRef.current) clearInterval(loopTimerRef.current); };
  }, [smartAutoSnap, session, samplesCount, cameraActive, evaluateOrCaptureFrame]);

  const progressPercent = Math.min(100, Math.round((samplesCount / TOTAL_SAMPLES_TARGET) * 100));
  const currentStepDef = POSE_DEFINITIONS[Math.min(samplesCount, TOTAL_SAMPLES_TARGET - 1)];
  const phase = liveMetrics.phase;
  const phaseDisplay = PHASE_DISPLAY[phase] || PHASE_DISPLAY.guidance;

  // Hold/collect progress bar (during hold and collecting phases)
  const holdPct = liveMetrics.hold_required > 0
    ? Math.min(100, (liveMetrics.hold_progress / liveMetrics.hold_required) * 100)
    : 0;
  const collectPct = liveMetrics.collect_required > 0
    ? Math.min(100, (liveMetrics.collect_progress / liveMetrics.collect_required) * 100)
    : 0;

  return (
    <div className="page-container">
      {/* Top Header Card */}
      <div className="enrollment-header-card">
        <div className="enrollment-header-left">
          <label className="form-label" htmlFor="enroll-emp-select">Select Enrolling Employee</label>
          <div className="select-row">
            <select
              id="enroll-emp-select"
              className="select-field"
              value={selectedEmployee?.id || ''}
              onChange={(e) => {
                const emp = employees.find((x) => x.id === parseInt(e.target.value, 10));
                setSelectedEmployee(emp || null);
                setSession(null);
                setSamplesCount(0);
                setFeedback(null);
              }}
              disabled={Boolean(session && samplesCount < TOTAL_SAMPLES_TARGET && samplesCount > 0)}
            >
              {employees.map((emp) => (
                <option key={emp.id} value={emp.id}>
                  {emp.name} ({emp.employee_code}) {emp.department ? `— ${emp.department}` : ''}
                </option>
              ))}
            </select>

            {!session || samplesCount >= TOTAL_SAMPLES_TARGET ? (
              <button type="button" className="btn btn-primary" onClick={handleStartSession} disabled={!selectedEmployee}>
                {samplesCount >= TOTAL_SAMPLES_TARGET ? 'Re-enroll (Add Angles)' : 'Start Enrollment'}
              </button>
            ) : (
              <button type="button" className="btn btn-secondary" onClick={() => { setSession(null); setSamplesCount(0); setFeedback(null); }}>
                Reset Session
              </button>
            )}

            {selectedEmployee && (
              <button type="button" className="btn btn-danger btn-sm" onClick={handleResetEnrollment} title="Delete all embeddings and start fresh">
                🗑 Re-enroll (Clear All)
              </button>
            )}
          </div>
        </div>

        <div className="enrollment-header-right">
          <div className="progress-meta">
            <span className="progress-label">Multi-Angle Progress</span>
            <span className="progress-count">{samplesCount} / {TOTAL_SAMPLES_TARGET} Angles</span>
          </div>
          <div className="progress-bar-track">
            <div className="progress-bar-fill" style={{ width: `${progressPercent}%` }} />
          </div>
        </div>
      </div>

      {/* Main Grid View */}
      <div className="enrollment-main-grid">
        {/* Left: Camera + Phase Overlay */}
        <div className="enrollment-camera-card">
          <div className={`camera-frame-box ${flashActive ? 'camera-flash-active' : ''}`}>
            {cameraError ? (
              <div className="camera-error-box">
                <p>{cameraError}</p>
                <button type="button" className="btn btn-secondary btn-sm" onClick={startCamera}>
                  Retry Camera
                </button>
              </div>
            ) : (
              <div className="video-overlay-wrapper">
                <video ref={videoRef} className="enrollment-video" playsInline muted autoPlay />

                {/* ── Dynamic Biometric Face Outline Overlay ── */}
                <svg className="biometric-svg-overlay" viewBox="0 0 400 450" preserveAspectRatio="xMidYMid meet">
                  <defs>
                    {/* Dark translucent vignette outside the face oval */}
                    <mask id="faceMask">
                      <rect width="400" height="450" fill="white" />
                      <ellipse cx="200" cy="220" rx="100" ry="135" fill="black" />
                    </mask>
                    {/* Linear gradients for glow effects */}
                    <linearGradient id="bracketGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor={phaseDisplay.color} stopOpacity="1" />
                      <stop offset="100%" stopColor={phaseDisplay.color} stopOpacity="0.6" />
                    </linearGradient>
                  </defs>

                  {/* Darkened vignette around face */}
                  <rect width="400" height="450" fill="rgba(10, 15, 29, 0.42)" mask="url(#faceMask)" />

                  {/* Base Dotted Oval Track */}
                  <ellipse cx="200" cy="220" rx="100" ry="135" className="biometric-oval-track" />

                  {/* Active Glowing Biometric Oval */}
                  <ellipse
                    cx="200"
                    cy="220"
                    rx="100"
                    ry="135"
                    className={`biometric-oval-glow ${phase === 'holding' || evaluating ? 'biometric-pulse-circle' : ''}`}
                    stroke={phaseDisplay.color}
                  />

                  {/* Biometric Corner Brackets */}
                  <path d="M 85 120 L 85 100 A 15 15 0 0 1 100 85 L 120 85" className="biometric-bracket" stroke={phaseDisplay.color} />
                  <path d="M 280 85 L 300 85 A 15 15 0 0 1 315 100 L 315 120" className="biometric-bracket" stroke={phaseDisplay.color} />
                  <path d="M 85 320 L 85 340 A 15 15 0 0 0 100 355 L 120 355" className="biometric-bracket" stroke={phaseDisplay.color} />
                  <path d="M 280 355 L 300 355 A 15 15 0 0 0 315 340 L 315 320" className="biometric-bracket" stroke={phaseDisplay.color} />

                  {/* Eye Level Alignment Guide */}
                  <line x1="120" y1="180" x2="280" y2="180" className="biometric-guideline" />
                  <text x="200" y="172" className="biometric-guide-text">ALIGN EYES HERE</text>

                  {/* Chin Level Alignment Guide */}
                  <line x1="160" y1="340" x2="240" y2="340" className="biometric-guideline" />
                  <text x="200" y="365" className="biometric-guide-text">CHIN LEVEL</text>

                  {/* ── Dynamic Directional Visual Cues based on currentPose ── */}
                  {session && currentPose === 'slight_left' && (
                    <g className="anim-left" style={{ color: '#38bdf8' }}>
                      <circle cx="85" cy="220" r="18" fill="rgba(56, 189, 248, 0.25)" stroke="#38bdf8" strokeWidth="2" />
                      <path d="M 92 220 L 78 220 M 84 214 L 78 220 L 84 226" stroke="#38bdf8" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                      <text x="85" y="255" fill="#38bdf8" fontSize="11" fontWeight="700" textAnchor="middle">TURN LEFT</text>
                    </g>
                  )}

                  {session && currentPose === 'slight_right' && (
                    <g className="anim-right" style={{ color: '#38bdf8' }}>
                      <circle cx="315" cy="220" r="18" fill="rgba(56, 189, 248, 0.25)" stroke="#38bdf8" strokeWidth="2" />
                      <path d="M 308 220 L 322 220 M 316 214 L 322 220 L 316 226" stroke="#38bdf8" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                      <text x="315" y="255" fill="#38bdf8" fontSize="11" fontWeight="700" textAnchor="middle">TURN RIGHT</text>
                    </g>
                  )}

                  {session && currentPose === 'slight_up' && (
                    <g className="anim-up" style={{ color: '#38bdf8' }}>
                      <circle cx="200" cy="95" r="18" fill="rgba(56, 189, 248, 0.25)" stroke="#38bdf8" strokeWidth="2" />
                      <path d="M 200 102 L 200 88 M 194 94 L 200 88 L 206 94" stroke="#38bdf8" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                      <text x="200" y="70" fill="#38bdf8" fontSize="11" fontWeight="700" textAnchor="middle">TILT CHIN UP</text>
                    </g>
                  )}

                  {session && currentPose === 'slight_down' && (
                    <g className="anim-down" style={{ color: '#38bdf8' }}>
                      <circle cx="200" cy="345" r="18" fill="rgba(56, 189, 248, 0.25)" stroke="#38bdf8" strokeWidth="2" />
                      <path d="M 200 338 L 200 352 M 194 346 L 200 352 L 206 346" stroke="#38bdf8" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                      <text x="200" y="380" fill="#38bdf8" fontSize="11" fontWeight="700" textAnchor="middle">TILT CHIN DOWN</text>
                    </g>
                  )}

                  {session && currentPose === 'smile' && (
                    <g style={{ color: '#f59e0b' }}>
                      <path d="M 175 285 Q 200 310 225 285" fill="none" stroke="#f59e0b" strokeWidth="3" strokeLinecap="round" />
                      <text x="200" y="272" fill="#f59e0b" fontSize="11" fontWeight="700" textAnchor="middle">SMILE NATURALLY</text>
                    </g>
                  )}

                  {session && currentPose === 'straight' && (
                    <g style={{ color: phaseDisplay.color }}>
                      <circle cx="200" cy="220" r="6" fill="none" stroke={phaseDisplay.color} strokeWidth="1.5" />
                      <line x1="200" y1="208" x2="200" y2="232" stroke={phaseDisplay.color} strokeWidth="1.5" />
                      <line x1="188" y1="220" x2="212" y2="220" stroke={phaseDisplay.color} strokeWidth="1.5" />
                    </g>
                  )}
                </svg>

                {/* Phase Status Banner */}
                {session && samplesCount < TOTAL_SAMPLES_TARGET && (
                  <div className="angle-overlay-banner" style={{ borderColor: phaseDisplay.color }}>
                    <span className="angle-icon">{currentStepDef.icon}</span>
                    <span className="angle-text" style={{ color: phaseDisplay.color }}>
                      {liveMetrics.guidance || currentStepDef.name}
                    </span>
                  </div>
                )}

                {/* Hold progress bar (phase: holding) */}
                {session && phase === 'holding' && liveMetrics.hold_required > 0 && (
                  <div className="phase-progress-bar-wrap" style={{ background: '#1e293b' }}>
                    <div className="phase-progress-label">Hold still {liveMetrics.hold_progress}/{liveMetrics.hold_required}</div>
                    <div className="phase-progress-track">
                      <div className="phase-progress-fill" style={{ width: `${holdPct}%`, background: '#f59e0b' }} />
                    </div>
                  </div>
                )}

                {/* Collect progress bar (phase: collecting) */}
                {session && phase === 'collecting' && liveMetrics.collect_required > 0 && (
                  <div className="phase-progress-bar-wrap" style={{ background: '#1e293b' }}>
                    <div className="phase-progress-label">Capturing best frame {liveMetrics.collect_progress}/{liveMetrics.collect_required}</div>
                    <div className="phase-progress-track">
                      <div className="phase-progress-fill" style={{ width: `${collectPct}%`, background: '#6366f1' }} />
                    </div>
                  </div>
                )}

                {/* Shutter Flash */}
                {flashActive && <div className="shutter-flash-overlay" />}
                <canvas ref={canvasRef} style={{ display: 'none' }} />
              </div>
            )}
          </div>

          {/* Controls Strip */}
          <div className="camera-controls-strip">
            <div className="controls-left-group">
              <label className="toggle-switch-label">
                <input
                  type="checkbox"
                  checked={smartAutoSnap}
                  onChange={(e) => setSmartAutoSnap(e.target.checked)}
                  disabled={!session || samplesCount >= TOTAL_SAMPLES_TARGET}
                />
                <span className="toggle-text"><strong>Smart Auto-Snap</strong> (auto-captures on angle match)</span>
              </label>
            </div>

            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => evaluateOrCaptureFrame(true)}
              disabled={!session || samplesCount >= TOTAL_SAMPLES_TARGET || evaluating || !cameraActive}
            >
              Manual Force Snapshot
            </button>
          </div>

          {/* Angle Metrics Strip */}
          {session && (
            <div className="angle-radar-strip">
              <div className="radar-item">
                <span className="radar-label">Yaw (Horizontal):</span>
                <span className={`radar-val ${Math.abs(liveMetrics.yaw) > 10 ? 'radar-active' : ''}`}>
                  {liveMetrics.yaw ? `${liveMetrics.yaw > 0 ? '+' : ''}${liveMetrics.yaw.toFixed(1)}°` : '0.0°'}
                </span>
              </div>
              <div className="radar-item">
                <span className="radar-label">Pitch (Vertical):</span>
                <span className={`radar-val ${Math.abs(liveMetrics.pitch) > 10 ? 'radar-active' : ''}`}>
                  {liveMetrics.pitch ? `${liveMetrics.pitch > 0 ? '+' : ''}${liveMetrics.pitch.toFixed(1)}°` : '0.0°'}
                </span>
              </div>
              <div className="radar-item">
                <span className="radar-label">Phase:</span>
                <span className="radar-val radar-status" style={{ color: phaseDisplay.color }}>
                  {phaseDisplay.label}
                </span>
              </div>
              {liveMetrics.score_100 != null && (
                <div className="radar-item">
                  <span className="radar-label">Quality:</span>
                  <span className="radar-val">{liveMetrics.score_100.toFixed(1)}%</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: Guide Panel */}
        <div className="enrollment-guide-panel">
          {/* Current Target Pose */}
          <div className="panel-section">
            <h3 className="panel-title">Current Target Angle</h3>
            <div className="current-pose-card">
              <div className="pose-badge-row">
                <span className="pose-step-badge">
                  Angle {Math.min(samplesCount + 1, TOTAL_SAMPLES_TARGET)} of {TOTAL_SAMPLES_TARGET}
                </span>
                <StatusBadge
                  status={samplesCount >= TOTAL_SAMPLES_TARGET ? 'passed' : 'in_progress'}
                  label={samplesCount >= TOTAL_SAMPLES_TARGET ? 'Completed' : currentStepDef.name}
                />
              </div>
              <p className="pose-instruction-text">{instructions}</p>

              {/* Phase indicator */}
              {session && samplesCount < TOTAL_SAMPLES_TARGET && (
                <div className="phase-indicator" style={{ color: phaseDisplay.color, fontWeight: 600, marginTop: 8 }}>
                  {phaseDisplay.label}
                </div>
              )}
            </div>
          </div>

          {feedback && (
            <div className={`feedback-alert feedback-${feedback.type}`}>
              <span className="feedback-icon">{feedback.type === 'success' ? '✓' : '⚠️'}</span>
              <span className="feedback-text">{feedback.message}</span>
            </div>
          )}

          {/* Angle Checklist */}
          <div className="panel-section">
            <h3 className="panel-title">Required Angle Sequence</h3>
            <div className="angle-steps-list">
              {POSE_DEFINITIONS.map((p, idx) => {
                const isDone = samplesCount > idx;
                const isCurrent = samplesCount === idx && session;
                return (
                  <div
                    key={idx}
                    className={`angle-step-pill ${isDone ? 'step-done' : ''} ${isCurrent ? 'step-active' : ''}`}
                  >
                    <span className="step-icon">{isDone ? '✓' : p.icon}</span>
                    <span className="step-label">{p.name}</span>
                    <span className="step-status-tag">
                      {isDone
                        ? 'Captured'
                        : isCurrent
                          ? (phase === 'holding' ? 'Holding...' : phase === 'collecting' ? 'Capturing...' : 'Active')
                          : 'Pending'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Completion Card */}
          {samplesCount >= TOTAL_SAMPLES_TARGET && (
            <div className="enrollment-completed-box">
              <span className="completion-icon">🎉</span>
              <h4>Enrollment Complete!</h4>
              <p>All {TOTAL_SAMPLES_TARGET} quality-gated ArcFace embeddings have been indexed in FAISS.</p>
              <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginTop: 8 }}>
                Run <code>python backend/calibrate.py</code> to verify embedding quality and recommended thresholds.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
