"""
Attendance Service

Logic:
  - First recognition today → INSERT check_in
  - Subsequent recognitions same day → Ignore (no duplicate rows)
  - Check-out → EXPLICIT ONLY (via API call or UI button)
  - No automatic re-detection checkout
"""
