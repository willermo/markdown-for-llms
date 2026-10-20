# Prepare the local product

This guide covers the experimental package `markdown_for_llms`, distinct from the legacy `markdown-*` commands. Target: Linux x86_64, managed CPython 3.12.13, CPU, single user and local filesystem. The API process does not import Marker or torch; the worker uses the same interpreter of the venv for the isolated children.

Follow the [uv guide](uv-environment.md) for the origin of the managed interpreter, input inventory, run snapshot, sdist→wheel build and canonical reinstallation. Add the extras `--extra app --extra marker-cpu` to the sync; for development add `--group dev`. Do not use `marker-cpu` and `marker-cu126` together. Run the distribution checks with the `cpu` profile when the venv contains ML; the `api` profile rejects those dependencies. The diagnostics identify the ten legacy modules and the files of the new package, including static files, migration and model inventory. The sync alone does not replace S/B/I.

Use `uv export --locked --no-default-groups --extra app --extra marker-cpu --no-emit-project` to prepare the requirements of the external venv. Installations with hashes require the archives identified by the lock. EbookLib 0.18 is distributed as an sdist: a local wheel built from that sdist requires its own identity and provenance, it does not inherit the digest of the source archive. Do not modify the historical benchmark venvs to install the product.

Prepare explicitly the system Bubblewrap (`bwrap`) and the data of the Marker 1.10.2 and Surya 0.17.1 profile. The inventory [models.json](../../markdown_for_llms/models.json) identifies the observed bytes of the five checkpoints and two fonts of the profile: recognition/layout2025_09_23, detection2025_05_07, table/OCR-error2025_02_18, GoNotoCurrent and GoNotoCJK. You can reuse a local copy already acquired with verified provenance; set `MDF_MODEL_ROOT` to its root. The worker verifies the fifty files before inference. It does not download or repair models during the conversion.

Acquiring the weights is a separate preparation, from the official sources of the [Marker pin](https://github.com/datalab-to/marker/tree/v1.10.2) and of [Surya](https://github.com/datalab-to/surya/tree/v0.17.1). The package does not include weights. The conditions of the code and of the weights are distinct; the use tested here is local and experimental, without redistribution of the models. The inventory records observed hashes, not signatures or independent checksums published by the vendor.

For this run the data already acquired are stored locally in `temp/run-a003-fase1-engine-comparison/work/implementation-r001/marker-models-r001`. It is a working path ignored by Git, not a resource guaranteed by a clone. The new product venv is under `temp/run-a004-fase2-prodotto-utilizzabile/work/implementation-r001/venv`. Do not carry these personal paths into the exported manifest.

Start API and worker as in the [tutorial](../tutorials/first-local-pdf.md). `/health` answers liveness only; `/api/v1/readiness` reports a missing worker, an incomplete runtime or unavailable isolation. Bubblewrap must be able to create private namespaces: a permission error does not enable a fallback without isolation. In the development environment use the escalation provided by the tools, keeping the barrier; do not modify sysctl, setuid or persistent profiles for the test.

The default archive quota is 20 GiB with 1 GiB of free reserve, besides the venv and the about 3.48 GB of checkpoints and fonts. The profile of the child has 16 GiB of maximum virtual memory per process and a sampled aggregate RSS guard; leave RAM for system/API/browser. The 450 s timeout is configurable and does not guarantee the conversion of 50 pages.

To change the quota or the timeout of an existing archive:

1. Stop API, worker and CLI that use the archive. A stop of the worker keeps the attempt as `interrupted`; the cancellation requested by the user is `cancelled`.
2. With the old environment file, read the `config` field of `identity.json`. The command also verifies roots, DB identity and schema; do not modify the JSON.
3. Run, replacing the digest with the one you read:

   ```bash
   mdf --env-file /percorso/prodotto.env configure --expected-config DIGEST --storage-mib 30720 --timeout-seconds 600
   ```

4. Update `MDF_STORAGE_MIB` and `MDF_TIMEOUT_SECONDS` in the same environment file and restart all the processes. The previous configuration is rejected. Manifests and configurations of the historical results stay unchanged.

The exclusive lock rejects the update if a service is still open. After a crash of the worker, restart it with the old configuration to reconcile children and reservations before the change. A wrong previous identity or a quota lower than the space occupied or committed blocks the operation. There is no need to delete the archive; general deletion stays outside I1.

If a result stays `publishing`, check `mdf show ID` or the detail in the browser. A quota/space rejection or an I/O/fsync fault keeps the complete result even if the rename has not happened and indicates the recovery of the same ID. A normal publication in progress needs no intervention; reading does not recompute the capacity of the archive at every update. If the recovery is deferred, restore the access or the free space of the volume without removing members of the archive; if a larger application quota is needed, use `configure` with all the services stopped. For this condition, the records of the dead worker may first require a start with the old configuration, which reconciles and releases them. Then restart the worker with the current configuration: it repeats checks and synchronizations before completing the publication. Check that the same ID becomes `completed` and that the ZIP is available. Do not use `retry` to recover `publishing`: that command creates a new conversion for a terminal execution.

An attempt that failed because of an unsafe entry keeps source and partial result; the other results and new submissions stay usable within the quota. You can create a new attempt with `mdf retry ID`; if the engine reproduces the unsafe entry it will fail again, explicitly. I1 offers no deletion or repair of partial results.
