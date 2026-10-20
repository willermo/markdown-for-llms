# ADR 0003 — SQLite and filesystem storage

- Date: 2026-10-02
- Decision status: accepted for the first local version
- Implementation status: I1 foundations r004 GO, integrated into dev at f8bf44c and push verified on 2026-10-17; revisions, backup and restore and further features remain to be built and proven
- Origin: the user's SQLite preference and accepted general approach

## Context and decision

Use SQLite for documents, executions, job states, effective configurations without secrets, detected problems, revisions and references to the artifacts. Keep source files, images, results and caches on the filesystem, separating durable storage from disposable caches. The file names supplied by the user are not identifiers.

Identify the sources by hash and assign distinct identities to the executions. Whether a result can be reused also depends on the configuration and on the versions of engines, models and pipeline. A new execution does not overwrite the previous one.

Publish a result only after the artifacts have been written and checked. Design explicitly the crash recovery and the reconciliation between DB and filesystem: there is no single implicit transaction covering both. Plan migrations, consistent backups and restore verification.

## Alternatives and consequences

SQLite reduces the services to administer in a single-host distribution. Use short transactions and measure the contention between API and worker. PostgreSQL becomes a choice to re-examine for several hosts or high write concurrency; it is not a requirement of the first version. Local DB and storage must reside on persistent volumes.

The [SQLite documentation](https://sqlite.org/whentouse.html) clarifies the concurrency limits and when a client/server database fits other scenarios. Operational link: [roadmap](../roadmap.md).
