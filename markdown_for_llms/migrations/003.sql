ALTER TABLE documents ADD COLUMN owner TEXT NOT NULL DEFAULT 'local';
ALTER TABLE runs ADD COLUMN owner TEXT NOT NULL DEFAULT 'local';
ALTER TABLE runs ADD COLUMN scope_recorded INTEGER CHECK(scope_recorded IN (0, 1));
CREATE TRIGGER runs_scope_recorded_monotonic BEFORE UPDATE OF scope_recorded ON runs
WHEN OLD.scope_recorded = 1 AND NEW.scope_recorded IS NOT 1
BEGIN SELECT RAISE(ABORT, 'scope_recorded is monotonic'); END;
PRAGMA user_version = 3;
