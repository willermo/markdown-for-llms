CREATE TABLE conversion_presets (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL, options TEXT NOT NULL,
    created REAL NOT NULL, updated REAL NOT NULL, UNIQUE(owner, name)
);
CREATE TABLE user_settings (owner TEXT PRIMARY KEY, default_preset TEXT, updated REAL NOT NULL);
PRAGMA user_version = 4;
