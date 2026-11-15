DROP TRIGGER runs_scope_recorded_monotonic;
ALTER TABLE runs DROP COLUMN scope_recorded;
ALTER TABLE runs DROP COLUMN owner;
ALTER TABLE documents DROP COLUMN owner;
PRAGMA user_version = 2;
