import React from 'react';

/**
 * Functional Status Badge — communicates operational states using restrained, clear tokens.
 */
export function StatusBadge({ status, label }) {
  const norm = String(status || '').toLowerCase();
  
  let variant = 'default';
  if (['active', 'running', 'started', 'passed', 'present', 'ok', 'true', 'online'].includes(norm)) {
    variant = 'success';
  } else if (['inactive', 'stopped', 'offline', 'error', 'failed', 'false', 'rejected'].includes(norm)) {
    variant = 'danger';
  } else if (['in_progress', 'evaluating', 'pending', 'warning', 'checked_out'].includes(norm)) {
    variant = 'warning';
  }

  const text = label || status;

  return (
    <span className={`status-badge status-${variant}`}>
      <span className="status-dot" />
      {text}
    </span>
  );
}
