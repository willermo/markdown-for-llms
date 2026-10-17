CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE documents (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, source_path TEXT NOT NULL UNIQUE,
    source_sha256 TEXT NOT NULL, source_bytes INTEGER NOT NULL, created REAL NOT NULL
);
CREATE TABLE runs (
    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
    retry_of TEXT REFERENCES runs(id), attempt_id TEXT NOT NULL UNIQUE,
    idempotency_key TEXT UNIQUE, options TEXT NOT NULL,
    state TEXT NOT NULL, bundle TEXT NOT NULL DEFAULT 'pending',
    quality TEXT NOT NULL DEFAULT 'unverified' CHECK(quality='unverified'),
    created REAL NOT NULL, updated REAL NOT NULL, token TEXT,
    lease REAL, child TEXT, cancel INTEGER NOT NULL DEFAULT 0,
    result_path TEXT, manifest TEXT, error TEXT
);
CREATE TABLE worker (id INTEGER PRIMARY KEY CHECK(id=1), token TEXT, heartbeat REAL, readiness TEXT);
CREATE INDEX runs_queue ON runs(state, created);
PRAGMA user_version = 1;
