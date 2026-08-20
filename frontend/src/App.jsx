import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { DashboardPage } from './pages/DashboardPage';
import { LiveRecognitionPage } from './pages/LiveRecognitionPage';
import { EmployeesPage } from './pages/EmployeesPage';
import { EnrollmentPage } from './pages/EnrollmentPage';
import { AttendancePage } from './pages/AttendancePage';
import { systemApi } from './api/system';
import { recognitionApi } from './api/recognition';
import './App.css';

export function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [backendOnline, setBackendOnline] = useState(false);
  const [engineRunning, setEngineRunning] = useState(false);
  const [currentFps, setCurrentFps] = useState(0);
  const [preselectedEmployee, setPreselectedEmployee] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  // Poll system health and recognition engine status
  const checkStatus = async () => {
    try {
      await systemApi.health();
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
    }

    try {
      const eng = await recognitionApi.getStatus();
      setEngineRunning(Boolean(eng.running));
      setCurrentFps(eng.current_fps || 0);
    } catch {
      setEngineRunning(false);
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await checkStatus();
    setRefreshKey((k) => k + 1);
    setTimeout(() => setIsRefreshing(false), 500);
  };

  const handleNavigateToEnrollment = (employee) => {
    setPreselectedEmployee(employee);
    setActiveTab('enrollment');
  };

  const PAGE_TITLES = {
    dashboard: { title: 'Operational Dashboard', subtitle: 'Face recognition attendance overview' },
    recognition: { title: 'Live Recognition', subtitle: 'Annotated 720p webcam feed & engine controls' },
    employees: { title: 'Employee Directory', subtitle: 'Manage registered employees & vector mappings' },
    enrollment: { title: 'Guided Face Enrollment', subtitle: 'Multi-pose variation capture & quality validation' },
    attendance: { title: 'Attendance Logs', subtitle: 'Daily check-in logs & explicit check-out records' },
  };

  const currentMeta = PAGE_TITLES[activeTab] || { title: 'Attendance System', subtitle: '' };

  return (
    <div className="app-shell">
      <Sidebar
        activeTab={activeTab}
        onTabChange={(tab) => {
          if (tab !== 'enrollment') setPreselectedEmployee(null);
          setActiveTab(tab);
        }}
        backendOnline={backendOnline}
        engineRunning={engineRunning}
      />

      <div className="app-main-layout">
        <TopBar
          title={currentMeta.title}
          subtitle={currentMeta.subtitle}
          backendOnline={backendOnline}
          engineRunning={engineRunning}
          currentFps={currentFps}
          onRefresh={handleRefresh}
          isRefreshing={isRefreshing}
        />

        <main className="app-content-area" key={refreshKey}>
          {activeTab === 'dashboard' && <DashboardPage onNavigate={setActiveTab} />}
          {activeTab === 'recognition' && <LiveRecognitionPage />}
          {activeTab === 'employees' && (
            <EmployeesPage onNavigateToEnrollment={handleNavigateToEnrollment} />
          )}
          {activeTab === 'enrollment' && (
            <EnrollmentPage preselectedEmployee={preselectedEmployee} />
          )}
          {activeTab === 'attendance' && <AttendancePage />}
        </main>
      </div>
    </div>
  );
}

export default App;
