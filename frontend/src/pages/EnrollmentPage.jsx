import React, { useState, useEffect, useRef } from 'react';
import { enrollmentApi } from '../api/enrollment';
import { employeesApi } from '../api/employees';
import { StatusBadge } from '../components/StatusBadge';

const TOTAL_SAMPLES_TARGET = 7;

const POSE_GUIDES = {
  straight: 'Look straight directly into the camera lens with a neutral expression.',
  slight_left: 'Turn your head slightly to the left (~15 degrees).',
  slight_right: 'Turn your head slightly to the right (~15 degrees).',
  slight_up: 'Tilt your chin slightly upward (~10 degrees).',
  slight_down: 'Tilt your chin slightly downward (~10 degrees).',
  smile: 'Smile naturally towards the camera lens.',
  complete: 'All required face variation samples have been successfully captured and indexed.',
};

export function EnrollmentPage({ preselectedEmployee }) {
  const [employees, setEmployees] = useState([]);
  const [selectedEmployee, setSelectedEmployee] = useState(preselectedEmployee || null);
  const [session, setSession] = useState(null);
  const [samplesCount, setSamplesCount] = useState(0);
  const [currentPose, setCurrentPose] = useState('straight');
  const [instructions, setInstructions] = useState(POSE_GUIDES.straight);
  const [feedback, setFeedback] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [autoCapture, setAutoCapture] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const autoCaptureTimerRef = useRef(null);

  // Load employee list for selection
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

  // Start client webcam for capture
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
      if (autoCaptureTimerRef.current) clearInterval(autoCaptureTimerRef.current);
    };
  }, []);

  const handleStartSession = async () => {
    if (!selectedEmployee) return;
    setFeedback(null);
    setSamplesCount(0);
    try {
      const startRes = await enrollmentApi.start({
        employee_code: selectedEmployee.employee_code,
        name: selectedEmployee.name,
        department: selectedEmployee.department || null,
      });
      setSession(startRes);
      setCurrentPose(startRes.current_pose || 'straight');
      setInstructions(startRes.instructions || POSE_GUIDES.straight);
      setFeedback({ type: 'info', message: 'Enrollment session initiated. Position face in frame.' });
    } catch (err) {
      // If code already exists, start session state locally for employee ID
      setSession({
        employee_id: selectedEmployee.id,
        employee_code: selectedEmployee.employee_code,
        name: selectedEmployee.name,
        status: 'in_progress',
        current_pose: 'straight',
        instructions: POSE_GUIDES.straight,
      });
      setCurrentPose('straight');
      setInstructions(POSE_GUIDES.straight);
    }
  };

  const captureAndSendSample = async () => {
    if (!videoRef.current || !canvasRef.current || !session) return;
    setCapturing(true);

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get Base64 image
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
    const base64Data = dataUrl.split(',')[1];

    try {
      const res = await enrollmentApi.sendSample({
        employee_id: session.employee_id,
        employee_code: session.employee_code,
        name: session.name,
        image_base64: base64Data,
      });

      if (res.success) {
        const nextCount = samplesCount + 1;
        setSamplesCount(nextCount);

        let nextPose = 'straight';
        if (nextCount >= 2 && nextCount < 3) nextPose = 'slight_left';
        else if (nextCount >= 3 && nextCount < 4) nextPose = 'slight_right';
        else if (nextCount >= 4 && nextCount < 5) nextPose = 'slight_up';
        else if (nextCount >= 5 && nextCount < 6) nextPose = 'slight_down';
        else if (nextCount >= 6 && nextCount < 7) nextPose = 'smile';
        else if (nextCount >= 7) nextPose = 'complete';

        setCurrentPose(nextPose);
        setInstructions(POSE_GUIDES[nextPose] || 'Look at the camera.');
        setFeedback({
          type: 'success',
          message: `Sample accepted! Quality Score: ${(res.score * 100).toFixed(1)}% (${nextCount}/${TOTAL_SAMPLES_TARGET})`,
        });

        if (nextCount >= TOTAL_SAMPLES_TARGET) {
          setAutoCapture(false);
        }
      } else {
        const errorDetails = {
          face_too_small: 'Face is too far. Please move closer to the camera lens.',
          too_blurry: 'Motion blur detected. Hold still while capturing.',
          too_dark: 'Lighting is too dark. Increase ambient lighting.',
          excessive_yaw: 'Head is turned too far sideways. Adjust angle to ~15 degrees.',
          excessive_pitch: 'Head is tilted too far up/down. Keep face upright.',
          no_face_detected: 'No face detected in video frame. Ensure unobstructed view.',
          multiple_faces_detected: 'Multiple faces detected. Only the enrolling employee must be in frame.',
        };

        setFeedback({
          type: 'error',
          message: errorDetails[res.reason] || `Quality check rejected: ${res.reason}`,
        });
      }
    } catch (err) {
      setFeedback({ type: 'error', message: err.message || 'Sample transmission failed' });
    } finally {
      setCapturing(false);
    }
  };

  // Auto-capture interval
  useEffect(() => {
    if (autoCapture && session && samplesCount < TOTAL_SAMPLES_TARGET && !capturing) {
      autoCaptureTimerRef.current = setInterval(() => {
        captureAndSendSample();
      }, 1500);
    } else {
      if (autoCaptureTimerRef.current) clearInterval(autoCaptureTimerRef.current);
    }
    return () => {
      if (autoCaptureTimerRef.current) clearInterval(autoCaptureTimerRef.current);
    };
  }, [autoCapture, session, samplesCount, capturing]);

  const progressPercent = Math.min(100, Math.round((samplesCount / TOTAL_SAMPLES_TARGET) * 100));

  return (
    <div className="page-container">
      {/* Employee Selector Bar */}
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
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleStartSession}
                disabled={!selectedEmployee}
              >
                {samplesCount >= TOTAL_SAMPLES_TARGET ? 'Re-enroll Employee' : 'Start Enrollment'}
              </button>
            ) : (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  setSession(null);
                  setSamplesCount(0);
                  setFeedback(null);
                }}
              >
                Reset Session
              </button>
            )}
          </div>
        </div>

        <div className="enrollment-header-right">
          <div className="progress-meta">
            <span className="progress-label">Enrollment Progress</span>
            <span className="progress-count">{samplesCount} / {TOTAL_SAMPLES_TARGET} Samples</span>
          </div>
          <div className="progress-bar-track">
            <div className="progress-bar-fill" style={{ width: `${progressPercent}%` }} />
          </div>
        </div>
      </div>

      {/* Main Guided Viewport */}
      <div className="enrollment-main-grid">
        {/* Left: Webcam Capture Canvas */}
        <div className="enrollment-camera-card">
          <div className="camera-frame-box">
            {cameraError ? (
              <div className="camera-error-box">
                <p>{cameraError}</p>
                <button type="button" className="btn btn-secondary btn-sm" onClick={startCamera}>
                  Retry Camera Permission
                </button>
              </div>
            ) : (
              <div className="video-overlay-wrapper">
                <video ref={videoRef} className="enrollment-video" playsInline muted autoPlay />
                <div className="face-guide-oval" />
                <canvas ref={canvasRef} style={{ display: 'none' }} />
              </div>
            )}
          </div>

          <div className="camera-controls-strip">
            <button
              type="button"
              className="btn btn-primary"
              onClick={captureAndSendSample}
              disabled={!session || samplesCount >= TOTAL_SAMPLES_TARGET || capturing || !cameraActive}
            >
              {capturing ? 'Evaluating Sample...' : 'Capture Sample Frame'}
            </button>

            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={autoCapture}
                onChange={(e) => setAutoCapture(e.target.checked)}
                disabled={!session || samplesCount >= TOTAL_SAMPLES_TARGET || !cameraActive}
              />
              <span>Auto-capture every 1.5s</span>
            </label>
          </div>
        </div>

        {/* Right: Pose Guidance & Feedback */}
        <div className="enrollment-guide-panel">
          <div className="panel-section">
            <h3 className="panel-title">Current Guided Pose</h3>
            <div className="current-pose-card">
              <div className="pose-badge-row">
                <span className="pose-step-badge">Step {Math.min(samplesCount + 1, TOTAL_SAMPLES_TARGET)} of {TOTAL_SAMPLES_TARGET}</span>
                <StatusBadge
                  status={samplesCount >= TOTAL_SAMPLES_TARGET ? 'passed' : 'in_progress'}
                  label={samplesCount >= TOTAL_SAMPLES_TARGET ? 'Completed' : currentPose.toUpperCase()}
                />
              </div>
              <p className="pose-instruction-text">{instructions}</p>
            </div>
          </div>

          {feedback && (
            <div className={`feedback-alert feedback-${feedback.type}`}>
              <span className="feedback-icon">{feedback.type === 'success' ? '✓' : '⚠️'}</span>
              <span className="feedback-text">{feedback.message}</span>
            </div>
          )}

          <div className="panel-section">
            <h3 className="panel-title">Stricter Quality Requirements</h3>
            <div className="quality-req-list">
              <div className="quality-req-item">
                <span className="req-name">Minimum Face Size</span>
                <span className="req-val">120 × 120 px</span>
              </div>
              <div className="quality-req-item">
                <span className="req-name">Blur (Laplacian Var)</span>
                <span className="req-val">&ge; 80.0</span>
              </div>
              <div className="quality-req-item">
                <span className="req-name">Lighting Brightness</span>
                <span className="req-val">&ge; 60 / 255</span>
              </div>
              <div className="quality-req-item">
                <span className="req-name">Max Head Angles</span>
                <span className="req-val">Yaw &le; 35°, Pitch &le; 35°</span>
              </div>
              <div className="quality-req-item">
                <span className="req-name">FAISS Vector Index</span>
                <span className="req-val">IndexFlatIP 512-D</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
