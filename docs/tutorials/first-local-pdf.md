# Convert your first local PDF

For a person who uses the experimental product I1 on Linux x86_64 and CPU. You need the package installed with the extras `app,marker-cpu`, managed CPython 3.12.13, Bubblewrap and the prepared checkpoints/fonts. The [setup guide](../how-to/local-product-setup.md) explains these prerequisites. No remote uploads, AI review or GPU are provided.

Copy [.env.example](../../.env.example) to a working file and set absolute paths for `MDF_DATA_ROOT` and `MDF_MODEL_ROOT`. Choose a new data directory, private and on a local filesystem. Do not use an output directory of the legacy pipeline. If port 8765 is taken, choose another `MDF_PORT` before creating the storage. API, worker and CLI must read the same file and environment.

In two terminals, using the Python of the installed venv:

```bash
"$RUN_APP_PY" -I -B -m markdown_for_llms.cli --env-file /percorso/prodotto.env serve
"$RUN_APP_PY" -I -B -m markdown_for_llms.cli --env-file /percorso/prodotto.env worker
```

Open `http://127.0.0.1:8765` (or the configured port). Wait for «Worker locale pronto» (local worker ready), choose a PDF and press **Converti** (Convert). The queue shows phase and identifier; it does not show estimated percentages. Reloading the page keeps the selection. The archive stays available even when the worker is not running.

When the conversion finishes, read the **Markdown originale** (original Markdown). The panel is text only: HTML, links and LaTeX are not executed. For long texts use «Leggi porzione successiva» (read next portion); the counter shows when you have read everything. The gallery shows only PNG/JPEG images that passed the separate raster check. Download with **Scarica sorgente** (download source) and **Scarica Markdown e asset (ZIP)** (download Markdown and assets). The ZIP contains `source/document.pdf`, `result/document.md`, the images in their original relative positions and `manifest.json`.

Check the result against the PDF, especially formulas, numbers, order, tables and figures. «Terminata» (finished) and «bundle disponibile» (bundle available) do not mean verified fidelity. An `unchecked` result stays downloadable when the files are intact, with reasons and scope visible; a known missing or altered asset blocks the ZIP.

**Annulla** (Cancel) stops the queued work or the active process. **Ripeti** (Retry) creates a new execution with another identifier, keeping the source and the previous attempts. After a stop of the worker, restart it: it reconciles the publications and marks the conversions left without an owner as interrupted. Repetition is explicit.

To use the CLI on the same archive:

```bash
"$RUN_APP_PY" -I -B -m markdown_for_llms.cli --env-file /percorso/prodotto.env convert esempio.pdf --wait
"$RUN_APP_PY" -I -B -m markdown_for_llms.cli --env-file /percorso/prodotto.env jobs
"$RUN_APP_PY" -I -B -m markdown_for_llms.cli --env-file /percorso/prodotto.env download ID --output risultato.zip
```

`--wait` requires the separate worker to be already running. A batch with errors returns a nonzero exit; read the outcomes per document. Do not remove members of the DB or of the storage by hand to free space: I1 does not include selective deletion from the archive. [Reference](../reference/local-product.md), [limits](../explanation/experimental-product.md).
