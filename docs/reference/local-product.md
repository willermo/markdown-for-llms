# Reference of the local product I1

Administrative configuration: [.env.example](../../.env.example). Precedence: defaults → file selected with `--env-file` → environment → `--data-root` and `--model-root`. The file is never looked up implicitly. `MDF_` prefix, unknown names and invalid values rejected, no additional JSON configuration for the user. The identities of storage, schema and configuration are verified by CLI, API and worker. The initial schema is 1; future schemas are rejected. No destructive migrations are planned.

| MDF variables | Default | Meaning |
| --- | --- | --- |
| `DATA_ROOT`, `MODEL_ROOT` | `./mdf-data`, `./mdf-models` | Distinct directories, without symlinks |
| `HOST`, `PORT` | `127.0.0.1`, 8765 | IPv4 loopback only; Host localhost or 127.0.0.1 on the chosen port |
| `BIND_PROFILE`, `REVIEW_PROVIDER` | `marker-cpu-v1`, `disabled` | Only admitted profile, no AI review |
| `TIMEOUT_SECONDS` | 450 | Deadline of the child, visible error; not an SLA |
| `MEMORY_MIB`, `THREADS`, `PROCESS_LIMIT` | 16384, 4, 512 | RLIMIT_AS per process, sampled RSS of the group; CPU affinity and process limits |
| `UPLOAD_MIB`, `MAX_PAGES` | 50, 50 | Explicit rejection; no `page_range` cut |
| `OUTPUT_MIB`, `LOG_MIB` | 512, 10 | File size limit plus total scratch guard; limited log |
| `STORAGE_MIB`, `RESERVE_MIB` | 20480, 1024 | Archive quota and free space; no automatic deletion |
| `MAX_PIXELS`, `MAX_DIMENSION` | 25000000, 16000 | Limits before PNG/JPEG decoding; other formats excluded from the gallery |
| `POLL_SECONDS`, `LEASE_SECONDS`, `GRACE_SECONDS` | 1, 15, 5 | Worker poll, lease, grace before the forced stop |

All integers are positive and finite. The lease must be at least three times the polling. The conversion options are passed to the pin and recorded separating requests, native translation and effective values. `force_ocr` and `strip_existing_ocr` are booleans; the other parameters of the profile are fixed. The mapping `disable_image_extraction=false` uses the native field `extract_images=true`. Batch layout/recognition/detection/table/OCR-error 1, pdftext worker 1, CPU, DPI 96/192, interop 1, `use_llm=false`, `paginate_output=true`.

| `mdf` command | Function |
| --- | --- |
| `serve` | Loopback API/UI; no inference in the process |
| `worker [--once]` | Separate worker; a single ML ownership per storage |
| `convert FILE... [--wait] [--force-ocr]` | Separate submissions; optional wait for the worker; nonzero batch for errors |
| `configure --expected-config HASH [--storage-mib N] [--timeout-seconds N]` | Administrative change with the old configuration verified and the archive stopped |
| `jobs`, `show ID` | State, identity, problems, configuration and manifest |
| `retry ID` | New execution linked to the finished one |
| `cancel ID` | Cancels the queued job or requests the stop of the active processes |
| `download ID --output PATH [--overwrite]` | ZIP after identity check; overwriting only when explicit |

Every command accepts `--env-file`, `--data-root`, `--model-root` before the subcommand. Equivalent: `python -I -B -m markdown_for_llms.cli`. The CLI outcomes are JSON; the axes `state`, `bundle`, `quality` and the message stay distinct. A downloadable `unchecked` is a warning; `incomplete`/`invalid` are not successes of the batch.

API under `/api/v1`: `GET session`, `GET readiness`, `GET runs`, `GET runs/ID`, `POST runs`, `POST runs/ID/retry`, `POST runs/ID/cancel`, `GET runs/ID/text?offset=N`, `GET runs/ID/artifacts/ARTIFACT_ID`, `GET runs/ID/source`, `GET runs/ID/bundle`. Upload: binary PDF body, header `X-Filename` with percent-encoding and optional `Idempotency-Key`. Mutations: exact Origin and `X-MDF-CSRF` from the session. CORS closed; local OpenAPI JSON, no Swagger/CDN. Assets are addressed by ID, never by the path of the request. The text is paginated at 256 KiB with a byte offset aligned to UTF-8; `complete` signals that everything has been read. Negative offsets, offsets beyond the end or on continuation bytes return HTTP 400, without invalidating the bundle; the offset of the end of the document returns a complete empty portion. `next_offset` is always reusable. Really corrupted UTF-8 stays an integrity error and blocks the ZIP.

The manifest schema 1 includes document/run/attempt, source hash, inventory with ID/hash/bytes, profile/timings, reference check and limits. It excludes environment secrets and host paths. The JSON of the native model and the limited logs stay in the diagnostic directories of the attempts; they are not included automatically in the ordinary download. The raw output is not normalized or cleaned.

The data directory must belong to the current user and must not be writable by the group or by other users. The data and model roots cannot coincide or contain each other. The isolation protects from the imported documents and from the conversion children; it does not protect from another hostile process with the same account that modifies database, runtime or administrative directories directly.

The shared capacity includes sources, attempts, logs, metadata, SQLite, ZIP and staging. Every writer reserves the space before growing: upload up to the limit plus 64 KiB, worker `2 × OUTPUT_MIB + LOG_MIB + 1` MiB, publication with an additional margin for the DB. 256 KiB remain for the administrative check. The bytes already present in the reservation are not counted a second time. The free reserve of the volume is added to the commitments still to be written. Inference does not hold the admission lock or a DB transaction. Scratch size and RSS remain sampled checks; before publishing, quota and commitments are checked again. During the ticks only the own budget and the physical reserve are checked; the global scan happens at admission and at publication, outside inference. Engine/raster/manifest metadata have a limit of 1 MiB and a maximum depth of 128 containers, inventory 4096 entries. The limit includes the actual UTF-8 bytes and the final newline. The complete manifest is validated before `publishing`: the envelope adds depth to the engine metadata. New manifests in file, DB and ZIP share the same UTF-8 serialization; the historical ones are not rewritten. NaN, Infinity and exponents that produce non-finite numbers are rejected. An exceeded limit produces a visible error.

The HTTP upload writes directly into the attributed staging, without a second copy. Rejections, interrupted streams and conflicts are collected; the accepted sources stay kept. A replay with a key already registered compares signature, size, hash and options by reading the stream without writing a copy of it, also when there is no space for a new upload.

The ZIP is a single immutable cache per attempt, with digest and CRC check at every delivery; all the originals and the manifest are verified also when the cache exists. It uses ZIP_STORED: the space needed is the sum of the members, of the manifest and of the headers, with a 64 KiB margin. Concurrent creations of the same ZIP are serialized; repeated downloads and disconnections reuse the file. The `.part` files belong to a reservation and are removed on error or in the recovery after a crash. No collection based only on the suffix; live writers are protected by flock. The cache counts in the quota, without a new copy at every click.

A missing, non-regular, unsafe or altered file invalidates the bundle. `EMFILE`, `EIO`, `ENOMEM` and transient permissions make the reading unavailable: HTTP 503 with `retryable=true`, without permanent invalidation. After the recovery the same result is retried. SQLite errors are explicit; the worker limits the recovery attempts and, with a DB that cannot be written, ends declaring the reconciliation needed at restart, without inventing a persisted state.

A capacity, reading or synchronization error during the recovery leaves `publishing` recoverable: detail and list expose `error` and `recovery`. The reading shows the last recorded reason, when available, without global scan or admission lock. If the quota prevents recording the diagnosis, a generic reason of deferred publication appears when the worker does not hold its lock; no diagnostic write beyond the quota. If it cannot record the diagnosis, the worker ends releasing the ownership: the deferred state stays observable without a capacity check on reading. An active publication without errors does not propose a restart automatically. The restart of the worker attempts a finite reconciliation of the same ID, without new inference. Before `completed` it repeats the fsync of the files and tree and of the parents `attempts`/`results`, also when the rename has already been done. Readable hashes do not attest durability by themselves. A persistent fault does not enable the download. The complete result can still be in `attempts` after an error before the rename, or already in `results`: both locations are recoverable. Both present or both absent, an inconsistent manifest or altered files produce an explicit integrity error, keeping the remaining bytes. An incomplete scratch in `running` instead stays `interrupted` with a distinct retry. Before the capacity check the recovery collects the identified children of the dead worker and releases their commitments; the live owners stay protected.

Symlink/hardlink/FIFO/socket entries in the scratch block the attempt. After the failure and the collection of the children, the count of the attributed partial results in the DB includes the unsafe entries without following or opening them (at least 4 KiB per special entry); other submit/ZIP/retry operations can proceed within the quota. Unattributed paths, sources, published results, DB and runtime stay strictly checked. No automatic removal of the partial result.

A corrupted ZIP cache or one without its receipt stays unavailable; I1 does not include a rebuild or removal command. The guards are sampled: the WAL growth and the consumption of other programs are not a kernel quota. The final check prevents `completed` when the capacity is exceeded.
