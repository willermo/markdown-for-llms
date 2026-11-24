ALTER TABLE runs DROP COLUMN gpu;
DROP TABLE gpu_requests;
DROP TABLE gpu_sessions;
PRAGMA user_version = 4;
