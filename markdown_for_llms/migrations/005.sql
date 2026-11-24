CREATE TABLE gpu_sessions (
    arm_id TEXT PRIMARY KEY, owner TEXT NOT NULL, gpu_preset TEXT NOT NULL, provider TEXT NOT NULL,
    resource TEXT NOT NULL, url TEXT NOT NULL, ledger_path TEXT NOT NULL, env_file TEXT, key_env TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('job', 'ahead')), a_cents TEXT NOT NULL, w_s INTEGER NOT NULL,
    t_arm REAL, until REAL,
    state TEXT NOT NULL CHECK (state IN ('opening', 'open', 'linger', 'closing', 'closed', 'uncertain', 'orphan',
                                         'stopped')),
    reason TEXT, t_zero REAL, limit_at REAL, drift_s REAL, created REAL NOT NULL, updated REAL NOT NULL
);
CREATE TABLE gpu_requests (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, gpu_preset TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('arm', 'disarm')),
    state TEXT NOT NULL CHECK (state IN ('pending', 'done', 'refused')), code TEXT,
    created REAL NOT NULL, updated REAL NOT NULL
);
ALTER TABLE runs ADD COLUMN gpu TEXT;
PRAGMA user_version = 5;
