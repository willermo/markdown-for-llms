# Convert documents with `mdf convert` and the archive of the product

For Linux users and developers. `mdf convert` converts one file directly with the conversion engine (`markdown_for_llms/engine/`), without the archive, the API or the worker: the output directory receives the complete Markdown document, its images and a machine-readable result. The archive of the product (the Web UI, the HTTP API and `mdf submit`, with the worker) converts with the same engine and keeps sources and results: see [Convert in the archive of the product](#convert-in-the-archive-of-the-product).

## Before you start

- The development venv of this repository (see [Prepare and update the uv environment](uv-environment.md)), with the package installed.
- Docker Engine on Linux with a local socket, and the toolchain image built once from a clean checkout. `mdf prepare --toolchain` is the same command as `mdf bench prepare --toolchain` ([Prepare the reference bench](bench.md)):

  ```bash
  .venv-dev/bin/mdf prepare --toolchain
  ```

- Bubblewrap at `/usr/bin/bwrap`: the audit of OOXML, OpenDocument and EPUB packages and the preparation of PDFs run in a child without network.
- For the pages family (PDF, images, DjVu) on the default route, Marker 2.2.0 on the CPU: the development venv has it (extra `engine`); its 18 weight files (indra-ocr v1.1.0 and the GoNoto font, about 111 MB) prepared once, and a systemd user manager (`systemd-run --user --scope`), which caps the memory of the Marker child:

  ```bash
  .venv-dev/bin/mdf prepare --weights
  ```

  `mdf prepare --weights` downloads each file by commit, verifies bytes and SHA-256, retries a few times, resumes from the files already verified and never overwrites one that does not verify. Every conversion verifies the 18 files again before it starts the child (`weights_missing`, `weights_mismatch`). With `mdf prepare --cpu-server`, which downloads the local CPU server of the `marker-server` route ([below](#convert-on-the-local-cpu-server)), it is the only command of the engine that downloads anything. The model root is `${XDG_CACHE_HOME:-~/.cache}/markdown-for-llms/models` unless you pass another one with `--model-root` or `MDF_MODEL_ROOT`, the same for `mdf prepare --weights`, `mdf convert` and the worker.

Nothing is downloaded during a conversion, and no document leaves the machine except on the Datalab route and on the `marker-server` route with a server outside the loopback, which you choose and confirm (below).

## Convert a file

```bash
.venv-dev/bin/mdf convert report.docx --out out/report
```

`--out` must be a new or empty directory. On success it holds:

| Path | Content |
| --- | --- |
| `document.md` | the document in Markdown (CommonMark with the pandoc extensions of `commonmark_x`), UTF-8 |
| `assets/` | the images the document references, named `asset-NNNN.<ext>` |
| `diagnostics/` | the extraction as the converter produced it, before the corrections of the engine (`pre-filter.md`, `marker-original.md`, `datalab-original.md`), and bounded logs |
| `engine.json` | the result: source and its SHA-256, route, tool steps, versions, inventory of every file with bytes and SHA-256, corrections and problems |

stdout carries one JSON summary (status, closed error code, route, number of files, problems); it never contains document text. On a terminal, stderr shows the steps. The exit status is 0 for a complete conversion, also when it has problems, 1 for a failure, 2 when the command is refused before converting (for example an output directory that is not empty).

A failed conversion leaves only `engine.json` (with `status: failed` and the closed code, for example `format_mismatch`) and `diagnostics/` with bounded logs; never a partial `document.md`.

## Read the problems

A complete conversion can carry problems, each with a severity:

- `incomplete`: content of the source is missing from the output. `media_lost` (an image the content uses that is not in the document, for example a picture inside an OpenDocument text box, an EPUB figure drawn by CSS, or an image the Markdown writer leaves out, such as an image nested in the alternative text of another inside a table written as HTML, also when its bytes are inline in the source), `media_missing` (an image whose source is not part of the document, for example an absolute path of a saved web page), `object_not_converted` (a chart, an OLE object or an inline SVG drawing, also an OpenDocument chart without a preview image in the document).
  On the pages routes: `page_text_lost` (a page with an OCR layer whose text OCR could not read).
- `warning`: nothing of the document's content is missing. `media_furniture` (an image of a slide layout, a header or a footer; on a PDF, raster cornice, see below), `media_unused` (a media file no part references), `media_external` (a remote image, left as a link and never fetched), `page_blank` (a page without text), `ocr_forced_document` (the whole document was read with OCR because an OCR layer could not be set aside), `pictures_skipped_rotated` (figures of a rotated page not extracted), `datalab_delete_failed`, `datalab_cost_unknown`.

Read them in `engine.json`: `problem_counts` (per severity) and `problem_codes` (per code) count all of them, `problems` lists at most 2,000, the `incomplete` ones first; the stdout summary carries the same counts. Links to targets outside the document (site paths, `javascript:`, files that are not part of the output, destinations that do not decode cleanly) are unlinked with their text kept, and active HTML is removed: these are recorded under `corrections`, not as problems.

## Choose a route

Each format has a default route; `--route` imposes one of its alternatives.

| Format | Default | Alternatives | In this version |
| --- | --- | --- | --- |
| docx, epub, html | `pandoc` | `marker` | available (`marker`: Marker's own provider makes a PDF first) |
| xlsx, pptx | `pandoc` | `marker`, `libreoffice` | available (`libreoffice`: see below) |
| odt, rtf | `pandoc` | — | available |
| doc, ods | `pandoc`, after LibreOffice converts them to docx and xlsx | — | available |
| txt | `text` (encoding from a byte order mark, otherwise detected by charset-normalizer; text kept verbatim) | — | available; see below for UTF-16 and UTF-32 |
| djvu | `marker` (after `ddjvu` makes a PDF of images) | `djvutxt`, `datalab`, `marker-server` | available; `djvutxt` keeps the text layer, page by page |
| pdf, png, jpeg, tiff | `marker` (an image becomes a PDF page, a TIFF one page per frame) | `datalab`, `marker-server` | available |

`marker-server` runs with `mdf convert` and `mdf bench run` on an inference server that you name, and in the archive of the product on the GPU presets of the instance, only with a `gpu_preset` (`route_not_in_product` otherwise, see [The marker-server route](#the-marker-server-route) and [The GPU session of the archive](#the-gpu-session-of-the-archive)).

```bash
.venv-dev/bin/mdf convert book.djvu --route djvutxt --max-pages 300 --out out/book
```

`--max-pages` (default `MDF_MAX_PAGES` of the product settings, 50) bounds the pages of the pages family (`page_limit`). `--lang` is recorded in `engine.json`; no converter of this version uses it. `--ocr force` reads every page with OCR on the Marker route and is refused with `route_option_unsupported` by the others. The memory cap of each tool (`MDF_MEMORY_MIB`, at most 2,048 MiB for the package audit), the output cap (`MDF_OUTPUT_MIB`, `engine.json` included, also the cap of the private working directory of each tool) and the total deadline (`MDF_TIMEOUT_SECONDS`, 0 for none) come from the product settings; `--env-file` of `mdf` selects them.

### The `libreoffice` route

LibreOffice, in the toolchain image, writes the workbook or the presentation as HTML, and pandoc reads that HTML with the same contract as the `pandoc` route:

- xlsx: Calc writes every sheet as a table and its images as files beside the HTML; those images are the images of the document.
- pptx: `impress_html_Export` writes one level-1 heading per slide (empty for a slide without a title) and its text, without images and without speaker notes. The images come from the package instead: the pictures of each slide, in the order of the slide list, follow the heading of their slide (`engine.json`: `libreoffice.images: package`, `grouped: slide`). The layout inside a slide is not rebuilt. When the headings are not as many as the slides, no image is placed and each one is `media_lost` with the detail `slide_boundaries_unknown`: an image never moves to another slide.

An export that LibreOffice leaves empty fails with `tool_failed`. `--ocr force` is refused.

```bash
.venv-dev/bin/mdf convert slides.pptx --route libreoffice --out out/slides
```

### Text files and their encoding

A txt that starts with a byte order mark is read in that encoding: UTF-32 (`FF FE 00 00`, `00 00 FE FF`), UTF-16 (`FF FE`, `FE FF`) or UTF-8 (`EF BB BF`), the 4-byte marks first. The Unicode text of Windows Notepad is UTF-16 with a mark. `engine.json` records the canonical name (`utf_32`, `utf_16`, `utf_8`) under `text.encoding` and `text.bom: true`. Without a mark charset-normalizer chooses the encoding; a short text in a legacy code page can be read in a neighbouring one.

After a UTF-16 or UTF-32 mark the whole file must decode strictly, and the text must not hold C0 control characters other than tab, line feed, vertical tab, form feed and carriage return: a file with any other one is refused with `format_mismatch`, at the upload too. This is a choice against binary files behind a mark, not a statement on the encoding, and it also refuses real texts: a UTF-16 or UTF-32 log with ANSI escapes (`ESC`, `\x1b`) or a text ending with the DOS end-of-file character (`\x1a`) is `format_mismatch`. Without a mark that rule does not apply (only a NUL byte is refused there). The UTF-8 form of the text must fit `MDF_OUTPUT_MIB` (`output_limit` otherwise, nothing written): from UTF-16 it can grow by half.

## The Marker route

Marker 2.2.0 runs in `fast` mode on the CPU, in a Bubblewrap child without network, the weights read-only, under a systemd scope whose memory cap is `MDF_MEMORY_MIB`; a child that writes no progress event and uses no CPU time for `MDF_STALL_SECONDS` is stopped (`stalled`). The PDF is classified first, page by page (`pages` in `engine.json`): `native` (visible text), `ocr_layer` (an image over most of the page with only invisible text, including the glyphless layers of OCR programs) or `image`.

- Only native pages: Marker reads their text; Indra, its reader, still runs OCR where the text layer is broken (for example a broken font map or text without ink) and says so per page: `indra_source` `hybrid` (part of the page) or `ocr` (the whole page), with the reason in `note`.
- No native page, or `--ocr force`: every page is read with OCR.
- Native pages with OCR-layer pages: the invisible text of those pages is removed and the copy is checked (the same rendering, no layer left); if the copy renders differently the original goes with OCR on every page (`ocr_forced_document`).

The times of the conversion are in `engine.json` under `timings`: `total_seconds` for every route and, on the Marker route, `stages` (seconds of loading the models, reading the pages, structure, processors and render, from the events of the child) and `pages` (the seconds of each page read by Indra, as the interval between two consecutive page events, with their median and maximum: the reading of the page only, not structure, processors and render; the seconds per page of the whole conversion are `total_seconds` divided by the pages). The archive keeps the stages and the statistics of the pages, not the value of each page.

Every page has its class, Indra's source and Marker's method in `engine.json`; an OCR-layer page keeps its class after the removal of the layer, the class of the clean copy is `class_after_strip`, and an OCR that finds nothing where the layer had text is `page_text_lost`. A native page that gives no text, an empty output or a missing page fail the conversion (`page_failed`, `output_empty`, `converter_failed`).

Figures of native pages: every raster image drawn on a page becomes an asset, at its place (an image entirely inside another one is part of its crop; two images that partly overlap make one crop of their union), except:

- an image of 1×1 pixel, a fill drawn as an image (a rule, a background): counted in `engine.json` (`rasters.fill`), no problem;
- cornice, reported once per group as `media_furniture`: the same image on two pages or more, always within the top or bottom 12 % of the page (a running header or footer); an image at the same place of that band on three pages or more, whatever its content (per-page codes of a footer); the same image on three pages or more elsewhere, whose first use stays an asset and only the repetitions are cornice (a figure repeated on consecutive slides is kept once).

Each page record counts its rasters by outcome (`picture`, `cornice`, `fill`, `page_image` for the image of a page read with OCR).

DjVu becomes a PDF of images before Marker, and that PDF is much larger than the DjVu (about 33 times on a 272-page book: 14 MB to 459 MB). For a long book raise `--max-pages` and `MDF_OUTPUT_MIB` together; otherwise the conversion stops with `output_limit`.

## The Datalab route

Datalab is a remote service: the document leaves the machine, and the conversion is paid. It runs only when you choose it, confirm the transit and set a spending cap:

```bash
.venv-dev/bin/mdf --env-file .env convert scan.pdf --route datalab --out out/scan \
    --datalab-mode balanced --datalab-cap-usd 0.05 --confirm-transit
```

- The key is `DATALAB_API_KEY` in the file given with `--env-file`, which must be a regular file of yours with mode 0600 (`datalab_key_unsafe` otherwise), or in the environment. It is never written to `engine.json`, the logs or stdout.
- Without `--confirm-transit` nothing is sent: the conversion stops with `transit_not_confirmed`, and the summary shows the service, the pages and the estimated cost (`cost`). The estimate (pages at the list price of the mode, plus one cent) above the cap stops it with `datalab_cap` before any byte leaves; the default cap is 0.
- What is sent is a PDF from which an invisible OCR layer has been removed; when it cannot be removed the route refuses (`ocr_layer_not_removable`). `--ocr force` is refused.
- The result is deleted from Datalab after every conversion, also after a failure or a cancellation; a failed deletion is the warning `datalab_delete_failed` (Datalab otherwise keeps it one day). The billed cost, when the answer gives it, is in `cost.billed_cents`. Whether Datalab may keep a copy for training depends on the settings of your Datalab team, not on this command.

## The marker-server route

`marker-server` runs Marker 2.2.0 in `balanced` or `accurate` mode with the OCR of Surya (the model `datalab-to/surya-ocr-2.5`) on an inference server that you name. It is a route of the pages family (pdf, png, jpeg, tiff, djvu) for `mdf convert` and `mdf bench run`; in the archive of the product it runs on the GPU presets of the instance, and is refused with `route_not_in_product` only without a `gpu_preset`. The Marker child still has no network: it reaches the server through a relay of the converting process, on a Unix socket in a private directory of yours, and it never sees a key.

- `balanced`: Marker reads the layout on this machine and the server reads the text lines; the tables are rebuilt on this machine from the lines that the server reads.
- `accurate`: every page goes to OCR, and the server reads the whole page, its tables included.

`--server-mode` is required with the route (`route_option_unsupported` without it). `--ocr force` makes `balanced` read every page with OCR too.

### Choose a server

| `kind` | The server | The page images leave the machine | Cost |
| --- | --- | --- | --- |
| `cpu` | the local CPU server: `llama-server` of llama.cpp b12040 with the GGUF weights of Surya, prepared once with `mdf prepare --cpu-server`, run in its own Bubblewrap child without network under a systemd user scope | no | none, but slow: 100 to 215 s for each page the server reads on the tested laptop (4 cores), about 6 s per page of text alone in `balanced` |
| `local` | a server of yours on the loopback: `http` to a literal loopback address with an explicit port, such as `http://127.0.0.1:PORT`, another `127.x.y.z` or `http://[::1]:PORT`; never a name such as `localhost` (`server_not_loopback` otherwise) | no | yours |
| `user` | a server of yours elsewhere, over https, for example vLLM 0.25.1 on your own GPU | yes | yours |
| `runpod`, `modal` | an endpoint rented by the second on RunPod or Modal, prepared with the scripts of `scripts/servers/` ([Prepare a RunPod endpoint](#prepare-a-runpod-endpoint), [Prepare a Modal app](#prepare-a-modal-app)) | yes | billed while a GPU runs: a spending cap, a ledger and a session are required |

The servers are named in `servers.json`, a file without secrets: `${XDG_CONFIG_HOME:-~/.config}/markdown-for-llms/servers.json`, or the file given with `--servers`. `default` names the server used without `--server`:

```json
{"default": "cpu", "servers": {
  "cpu": {"kind": "cpu"},
  "my-gpu": {"kind": "user", "url": "https://gpu.example.org", "key_env": ["MY_GPU_KEY"]},
  "runpod-24gb": {"kind": "runpod", "url": "https://<endpoint id>.api.runpod.ai", "gpu": "AMPERE_24",
                  "rate_usd_per_s": "0.0002090948", "pins": {"vllm": "0.25.1"}},
  "modal-l4": {"kind": "modal", "url": "https://<workspace>--mdf-surya-serve.<region>.modal.run", "gpu": "L4,A10G",
               "rate_usd_per_s": "0.00036772", "pins": {"vllm": "0.25.1"}}}}
```

The fields are a closed set (anything else is `server_not_configured`): `kind`, `provider` (a `user` server of RunPod or Modal costs like the endpoint, and needs the same session), `url`, `server` (`vllm` or `llamacpp`), `model`, `revision`, `pins` (the versions the server must declare at readiness, `server_mismatch` otherwise), `parallel`, `gpu`, `rate_usd_per_s` (required with a provider, and equal to the rate that the session records at its arm: 0.0002090948 for RunPod, 0.00036772 for Modal, the rate of the dearest GPU of the app; another value is `server_not_configured` before any request, so that a stale file never reserves less than the window costs), `rate_source`, `rate_date`, the seconds `ready_within`, `read_timeout`, `page_seconds`, `cold_start_s`, `w1_s`, `tail_idle_s`, `tail_fault_s`, and `key_env`, the names of the variables of its keys, each ending in `_KEY`, `_SECRET`, `_TOKEN`, `_TOKEN_ID` or `_BEARER` (the names `mdf bench secret-scan` searches; another name is `server_not_configured`). Without `key_env` a RunPod endpoint takes `RUNPOD_API_KEY` (`Authorization: Bearer`), a Modal endpoint `MODAL_PROXY_TOKEN_ID` and `MODAL_PROXY_TOKEN_SECRET` (`Modal-Key` and `Modal-Secret`), the others none. The keys come from the file given with `--env-file` of `mdf` (a regular file of yours with mode 0600, `server_key_unsafe` otherwise; `server_key_missing` without them) or from the environment; they go only into the headers of the relay and are never written. The relay asks every server for answers without compression (`Accept-Encoding: identity`); an answer compressed anyway (gzip or deflate) is decoded piece by piece within the limit of 30,000,000 bytes, and another encoding, a truncated answer or one with bytes after its end reaches Marker as an error (502).

### GPU presets

A GPU preset fixes where and how the inference runs on a rented GPU: provider, data center, volume (type, class and size), GPUs, image by digest, model with its revision and version of vLLM, rate with its source and date, the archives of the cache of compilation, whether the GPU can be armed ahead, the seconds of the terms, and the measures shown before a conversion with their date, hardware and source. The presets are in `markdown_for_llms/gpu/data/presets.json`, with a closed schema; another data center, volume or GPU is a new preset of that file, never a value written by hand:

| Preset | Status | Provider and place | Volume | GPU |
| --- | --- | --- | --- | --- |
| `modal-base` | supported | Modal, the region Modal chooses | the Volume `mdf-surya-weights` | L4, else A10G (A10 or A10G seen) |
| `runpod-serverless-volume-dc` | supported | RunPod serverless, CA-MTL-1 | network volume `HIGH_PERFORMANCE`, 10 GB | `AMPERE_24` (RTX A5000 seen) |
| `runpod-serverless-volume-eu-se-1` | superseded by `runpod-serverless-volume-dc` | RunPod serverless, EU-SE-1 | network volume `STANDARD`, 10 GB, deleted | `AMPERE_24` |

A superseded preset is listed for its measures and never used; a preset on a global volume is refused (`gpu_volume_unsupported`). An endpoint of `servers.json` names its preset with only five fields:

```json
{"default": "batch-gpu", "servers": {
  "batch-gpu": {"kind": "runpod", "preset": "runpod-serverless-volume-dc", "resource": "mdf-surya",
                "url": "https://<endpoint id>.api.runpod.ai"},
  "modal-gpu": {"kind": "modal", "preset": "modal-base", "resource": "mdf-surya",
                "url": "https://<workspace>--mdf-surya-serve.<region>.modal.run"}}}
```

`kind` is the provider of the preset, `resource` the name of the endpoint or app at the provider, `url` its address and `key_env`, optional, only the names that the provider of the preset reads, in their order (`RUNPOD_API_KEY`; `MODAL_PROXY_TOKEN_ID` and `MODAL_PROXY_TOKEN_SECRET`): the worker, its children and the relay read those, so other names are `server_not_configured`. The preset gives every other field (server, model, revision, pins, parallel, gpu, rate, seconds): one of them written next to `preset`, a preset of another provider or a superseded one, and two endpoints of one file with the same preset are `server_not_configured`. `mdf convert` and `mdf bench run` read such an endpoint as any other. For Modal the preset takes `ready_within` 900 and `cold_start_s` 480, and so does a Modal endpoint written by hand (with the former defaults, 1200 and 1100, `mdf convert` refused a reserve under about 0.53 USD).

### Convert on the local CPU server

```bash
.venv-dev/bin/mdf prepare --cpu-server
.venv-dev/bin/mdf convert scan.pdf --route marker-server --server-mode balanced --server cpu --out out/scan
```

`mdf prepare --cpu-server` downloads the release of `llama-server` and the two GGUF files, each by URL with its bytes and SHA-256, into `${XDG_CACHE_HOME:-~/.cache}/markdown-for-llms/cpu-server` (`--root` for another directory), and records whether that build listens on a Unix socket: without one the route cannot run in the sandbox. Every conversion verifies the files again before it starts the server.

### Convert on a server outside the loopback

```bash
.venv-dev/bin/mdf --env-file .env convert scan.pdf --route marker-server --server-mode accurate \
    --server my-gpu --confirm-transit --out out/scan
```

Without `--confirm-transit` nothing is sent: the conversion stops with `transit_not_confirmed` before the server is woken. `engine.json` records the service and the host under `transit` (`left_machine` true from the first request), and the requests, their statuses, the readiness and the close of the relay under `marker_server`. A server that is not ready within its `ready_within`, or that goes down, is `server_unavailable`; one that refuses the key at readiness (401, 403) `server_rejected`; one whose answers make Marker fail `server_failed`; the end of the term of the attempt `server_deadline`.

### Convert on a rented endpoint

A rented endpoint costs while a GPU runs. `mdf convert` sends pages to it only inside a session of `scripts/servers/session.py run`, which arms the endpoint for a window, runs your command, then disarms the endpoint and waits until nothing runs, whatever happened:

```bash
L=~/bench-runs/spend.jsonl
.venv-dev/bin/python -I -B scripts/servers/session.py run --provider modal --name mdf-surya \
    --url "https://<workspace>--mdf-surya-serve.<region>.modal.run" --env-file .env \
    --spend-ledger "$L" --spend-cap-usd 15 --cap-usd 1.00 -- \
  .venv-dev/bin/mdf --env-file .env convert report.pdf --route marker-server --server-mode balanced \
    --server modal-l4 --confirm-transit --server-cap-usd 1.00 --spend-ledger "$L" --spend-cap-usd 15 --out out/report
```

- `--cap-usd` is the cap of this session, `--spend-cap-usd` the cap of the ledger for that provider, all sessions together. The session takes `A = min(--cap-usd, what the ledger leaves)` and a window `W = A / rate - 60 s - Z - tail` (Z = 120 s for the zero after the disarm; tail = 30 s on RunPod, which bills until the container has stopped). It refuses to start when the ledger has an orphan window of the provider (`session_orphaned`, below) or when the provider is not idle (`provider_not_idle`).
- On Modal `--url` is the URL of the app that the arm records, so that `mdf convert` finds the session; on RunPod the session reads it from the endpoint.
- Inside the window, `mdf convert` needs `--server-cap-usd`, `--spend-ledger` and `--spend-cap-usd` (`server_ledger_required`), reserves the term of the attempt in the ledger before it wakes the server, and refuses with `server_not_armed` outside a session or when the window has less time left than the readiness floor of the endpoint (`ready_within`: 900 s on RunPod and on Modal by default), with `server_cap` when the ledger is over its cap.
- At the end of the command, at the end of the window, or on SIGINT or SIGTERM the session stops the command, disarms the endpoint and reads it every 10 s until nothing runs. The end of the window holds during the arm too: each call that arms the provider gets at most the time left to it, SIGINT, SIGTERM and the end of the window interrupt an arm in progress, and the command starts only while the window is open (exit 5 otherwise). An arm interrupted or finished after the end of the window may have changed the provider: the session disarms it and reads the zero as always. A window is clean when its command was `mdf convert` or `mdf bench` (also as `python -I -B -m markdown_for_llms.cli convert` or `bench`, after the options of `mdf`), ended with 0 and without a fault, and the zero came within Z: the ledger then counts the measured bounds of its attempts (`server_cost_bound` in the problems: an upper bound, not a bill); otherwise it counts the time from the arm to the zero (on RunPod to the stop of the container) plus the tail. No zero within 600 s leaves the window orphan at its cap: `session.py status` names it, `session.py disarm --provider P --name N --spend-ledger L --env-file .env` disarms again and closes it, and only a reconciliation with the bill releases its amount ([The ledger and the bill](#the-ledger-and-the-bill)).

The same session runs a bench (`mdf bench run --route marker-server`, see [the bench guide](bench.md#measure-the-marker-server-route)).

### The default server for each use

Measured on 21 November 2026 with vLLM 0.25.1 on the standard and quick levels of the reference set ([the bench guide](bench.md)), the cache of compilation in use where it says so:

| | Modal: conversions now (the default) | RunPod: batches (the default) |
| --- | --- | --- |
| GPU | NVIDIA A10 or A10G, 24 GB, as Modal assigns them | RTX A5000, 24 GB (pool `AMPERE_24`), Secure Cloud |
| Rate of the ledger | 0.00036772 USD/s (about 1.32 USD/h: the GPU, 2 cores, 16 GiB) | 0.0002090948 USD/s (about 0.75 USD/h with the disk) |
| Ready, from the first request | 164-172 s on an A10 with the cache, 221 s on an A10G with it; 260-273 s on an A10G without it | 163-167 s with the cache when the host has the image; 541-718 s, with or without the cache, when the host loads the image first; 192-201 s armed 420 s ahead, without the cache ([Arm ahead](#arm-ahead)) |
| Seconds per page after readiness | 1.76-1.84 (standard level, about 200 pages per mode); 0.36-0.39 (a scanned book of 272 pages) | 2.60-2.96 (quick level); 0.91 (a manual of 132 pages) |
| Cost per page after readiness | 0.065-0.068 US cents (standard); 0.013-0.014 cents (the scanned book) | 0.054-0.062 US cents (quick) |
| Billed for one start | 6.47 cents for a 7-page document with its start and the 60 s of idle | 3.07 cents for the same document; loading the image is not billed |

Modal starts a container in seconds and is ready in about 3 minutes, so a single document waits least there. RunPod costs 0.57 times as much per second, but a host that has not loaded the image yet takes 9 to 12 minutes, so it suits a batch, where one start serves many pages; arming ahead hides most of that wait. The local CPU server costs nothing and fits a few pages.

### Prepare a RunPod endpoint

The scripts of `scripts/servers/` run from the root of a clone with the development venv; they are not part of the package. They use the REST v2 API of RunPod (GraphQL only to read the balance), and every reading they make is saved in their output.

1. Create an API key restricted to what the scripts need (RunPod console, API keys): REST serverless read and write, network volumes read and write, billing read, catalog read, account read, nothing else; GraphQL read only (read and write would control the whole account); the endpoints' API (`api.runpod.ai`) read and write. Put it in your key file as `RUNPOD_API_KEY` (see [`.env.example`](../../.env.example)). The same key serves the scripts and the inference requests.
2. Choose a data center of Secure Cloud where the 24 GB pool has network volumes:

   ```bash
   .venv-dev/bin/python -I -B scripts/servers/runpod_endpoint.py --env-file .env catalog
   ```

   It lists the GPUs of the pool with `secure` and their data centers, the data centers with network volumes, and the balance.
3. Create the network volume `mdf-surya-weights` (10 GB) and the endpoint in that data center. `--dry-run` prints the bodies and creates nothing:

   ```bash
   .venv-dev/bin/python -I -B scripts/servers/runpod_endpoint.py --env-file .env create --name mdf-surya --data-center DC --dry-run
   .venv-dev/bin/python -I -B scripts/servers/runpod_endpoint.py --env-file .env create --name mdf-surya --data-center DC
   ```

   The endpoint is load-balanced, born disarmed (`workers.max` 0, `idleTimeout` 60 s), runs the vLLM image pinned by digest with a watchdog in front of vLLM, has FlashBoot off and is limited to the data center of its volume. The REST API has no field for the cloud of a serverless endpoint: it runs where its volume is, so check in the console that it shows Secure Cloud. `create` prints the endpoint id: the URL of `servers.json` is `https://<endpoint id>.api.runpod.ai`.
4. Check the pins and bring a disarmed endpoint back to them when they differ:

   ```bash
   .venv-dev/bin/python -I -B scripts/servers/runpod_endpoint.py --env-file .env verify-pins --name mdf-surya
   .venv-dev/bin/python -I -B scripts/servers/runpod_endpoint.py --env-file .env align --name mdf-surya
   ```

   `align` changes only the env map with the watchdog, the `workers` block, the CUDA floor and FlashBoot, which it turns off. The other pins (image, entrypoint, type, scaling, ports, the volume and its data center, the data center in the catalog with every GPU of the pool Secure, the pool and its serverless price up to 0.75 USD per hour) need a new endpoint; `verify-pins` names them in `outside_align`. Every session checks the same pins before its arm and refuses an endpoint outside them (`endpoint_out_of_pins`, exit 3) without changing it, and every arm checks them again on its own reading before its PATCH.
5. Load the weights on the volume once: a session with `--allow-download` runs the subcommand `populate` of `runpod_endpoint.py` (after `--`, the session runs these subcommands on its endpoint), which waits for vLLM to be ready with the pinned revision downloaded and verified:

   ```bash
   .venv-dev/bin/python -I -B scripts/servers/session.py run --provider runpod --name mdf-surya --allow-download \
       --env-file .env --spend-ledger ~/bench-runs/spend.jsonl --spend-cap-usd 4 --cap-usd 0.60 -- populate
   ```

6. Optionally, the cache of compilation ([below](#the-cache-of-compilation)).

`runpod_endpoint.py delete --name N` deletes an endpoint and, by the public documentation of the REST v2, its template with it; the command deletes the template apart only when the reading of the endpoint gives a `templateId`, and a `403` there is the declared outcome `template: forbidden` (exit 0: the endpoint is gone). The volume stays until you delete it in the console.

### Prepare a Modal app

1. Give the pinned SDK its token: `uv tool run --from modal==1.6.3 modal token new` opens a login in the browser and writes the token of the SDK to `~/.modal.toml`. `uv tool run --from modal==1.6.3 modal profile current` must name the workspace that pays; on its dashboard set the spending limit and the usage budget you accept.
2. Create a proxy token of the same workspace (Settings, Proxy tokens, or `uv tool run --from modal==1.6.3 modal workspace proxy-tokens create`, which prints the secret once) and put it in your key file in three lines: `MODAL_PROXY_TOKEN_ID` (it starts with `wk-`), `MODAL_PROXY_TOKEN_SECRET` (`ws-`) and `MODAL_PROXY_BEARER`, the two joined by a dot, for a `user` server that sends `Authorization: Bearer`.
3. Fill the Volume `mdf-surya-weights` once, in a session at the rate of a CPU:

   ```bash
   .venv-dev/bin/python -I -B scripts/servers/session.py run --provider modal --name mdf-surya --cpu-activity \
       --env-file .env --spend-ledger ~/bench-runs/spend.jsonl --spend-cap-usd 15 --cap-usd 0.10 -- \
     uv tool run --from modal==1.6.3 modal run scripts/servers/modal_populate.py::populate
   ```

   It downloads the pinned revision on a CPU container, checks the size and SHA-256 of the weights and commits the Volume. A CPU activity has no window: its `--cap-usd` is a reserve in the ledger, not a cap that stops the job on Modal. What stops the job is the timeout of the Function (1800 s, no retries), so `--cpu-activity` runs only this command and only with a reserve that covers that timeout at the CPU rate (1800 s × 0.00003508 USD/s = 6.32 cents): below it the session refuses before the command (`server_cap`), and any other command is `invalid_setting`. The reserve stays in the ledger until a manual reconciliation with the bill.
4. Each session deploys the app `mdf-surya` (`scripts/servers/modal_app.py`) at the arm and stops it at the disarm: an L4, else an A10G, at most one container, 60 s of idle before it scales to zero, requests only with the proxy token. The `armed` event of a session prints the URL of the app: run a first session with the command `true` (`… --cap-usd 0.10 -- true`: below 0.07 USD the window of Modal is not above 0 and the session refuses; no container starts without a request) and put that URL in `servers.json` and in `--url`.

### Two kinds of keys

Each provider needs a key that manages it (the sessions and the scripts of `scripts/servers/` use it) and a key that lets requests through to the endpoint (the relay of `mdf convert` and `mdf bench run` uses it). They are not interchangeable:

| | The key that manages the provider | The key that lets requests through | Where they are |
| --- | --- | --- | --- |
| Modal | the token of the SDK (`ak-…` and `as-…`): deploys, stops and reads the app, its containers, logs and bill, through the pinned SDK | the proxy token (`wk-…` and `ws-…`): it only opens the edge of the app's web endpoint, whose functions require it; a missing one, or one of another workspace, gets 401 from the edge and no request reaches the app | the token of the SDK in `~/.modal.toml`, never in the key file or given to `mdf`; the proxy token in the key file |
| RunPod | one restricted API key does both, so keep its permissions to the list above | the same key | the key file (`RUNPOD_API_KEY`) |
| Datalab | | `DATALAB_API_KEY`, for the Datalab route only | the key file |

### Revoke or restrict the keys when you stop

When you stop using a provider, delete its keys or reduce them, so that a copied file opens nothing: on RunPod delete the API key in the console (a key left with read permissions only still reads the endpoints, their logs and the bill of the account, so delete it unless you need that); on Modal revoke the token of the SDK and delete the proxy token in the settings of the workspace (`modal token new` creates a new one when you need it again); on Datalab rotate the key in the dashboard. First run [`mdf bench secret-scan`](bench.md#check-that-no-key-is-in-a-file) with your key file on the folders you keep, while the file still holds the values it searches; then remove the lines from the key file. Stop or delete what keeps costing: a RunPod network volume of type `STANDARD` costs 0.07 USD per GB and month (10 GB: 0.70 USD); in a data center that offers only `HIGH_PERFORMANCE` volumes, such as CA-MTL-1, `runpod_endpoint.py create` makes one of that type, which the console priced at 0.14 USD per GB and month on 22 November 2026 (10 GB: 1.40 USD), so check the price in the console before you create the volume; a Modal Volume 0.09 USD per GiB and month beyond what the plan includes; a disarmed endpoint and a stopped app cost nothing.

### The cache of compilation

vLLM compiles its model graphs at every start (`torch.compile`: in the starts of 21 November 2026, 28 to 36 s on an A10 or an RTX A5000 and 42 to 48 s on an A10G). A session can give it, instead, the compiled graphs of an earlier start, kept on the volume of the endpoint as an archive, under three conditions:

1. only an archive made by a populate of yours: the session with the lever `compile-cache-populate`, during which the watchdog refuses every inference (503 `mdf_populating`), so that no document reaches that container; the archive is written once vLLM is ready, before any request;
2. its SHA-256 checked before every use: the lever `compile-cache:<sha256>` names it, and an archive with another SHA-256 is not used;
3. no data of a document in it: only the compiled graphs, and a manifest with the GPU and the counts.

A use reads the archive within the limits of its populate: a regular file of at most 4 GiB of graphs plus the headers, only regular files and directories under the cache (no link, no device), extracted into a temporary directory that becomes the cache only whole; anything else, and vLLM compiles as without it.

```bash
.venv-dev/bin/python -I -B scripts/servers/session.py run --provider modal --name mdf-surya \
    --url "https://<workspace>--mdf-surya-serve.<region>.modal.run" --levers compile-cache-populate \
    --env-file .env --spend-ledger ~/bench-runs/spend.jsonl --spend-cap-usd 15 --cap-usd 0.40 -- \
  .venv-dev/bin/python -I -B scripts/servers/start_series.py --env-file .env --server modal-l4 \
    --servers ~/.config/markdown-for-llms/servers.json --input page.png --cold 1 --via populate \
    --spend-ledger ~/bench-runs/spend.jsonl --spend-cap-usd 15
```

The last line of `start_series.py` gives the SHA-256 of the archive (`sha256`). An archive serves only the GPU of its manifest: on another GPU vLLM compiles as without it. On Modal, which assigns an A10 or an A10G as it likes, `--gpu "NVIDIA A10G"` takes only a container on that GPU and lets the others go before vLLM is ready. Without `--levers` every session takes the archives adopted (`DEFAULT_LEVERS` of `markdown_for_llms/gpu/providers.py`), one per GPU: the A10 and the A10G of Modal and the RTX A5000 of RunPod in CA-MTL-1, `compile-cache:6222ea36…` (an L4 has none); `--levers none` arms without them. They are the archives of the supported GPU presets ([GPU presets](#gpu-presets)), `levers.compile_cache` of `markdown_for_llms/gpu/data/presets.json`. An archive lives on the volume of its endpoint: an endpoint on another volume, for example after a move to another data center, has none until a populate of its own, and `DEFAULT_LEVERS` names only the archives above, so name the new one with `--levers`. The archive of the volume of EU-SE-1, `compile-cache:d3818054…`, was deleted with that volume on 22 November 2026; it stays only in the superseded preset. A new image, another version of vLLM or other arguments of vLLM need a new populate: vLLM then does not find its graphs and compiles. With the cache, a start was ready 25 to 30 s earlier (`torch.compile` took 2.4 to 3.3 s), and every conversion of a one-page scan on 21 November 2026, more than 20 on the three GPUs, gave the same output with and without it.

### Arm ahead

On RunPod the arm alone makes a host load the image of the endpoint, and nothing is billed before a worker runs (checked on the bill). Arming about 7 minutes before the first request hides the load: in three sessions armed 420 s ahead the endpoint was ready 192 to 201 s after the first request, against 462 to 621 s after the first request without (both without the cache of compilation). In the archive, choose the preset `batch`, confirm the transit and press "Arm ahead" ([The GPU session of the archive](#the-gpu-session-of-the-archive)). From the command line, make the command of the session wait:

```bash
.venv-dev/bin/python -I -B scripts/servers/session.py run --provider runpod --name mdf-surya --env-file .env \
    --spend-ledger ~/bench-runs/spend.jsonl --spend-cap-usd 4 --cap-usd 0.40 -- \
  sh -c 'sleep 420 && exec .venv-dev/bin/mdf --env-file .env convert report.pdf --route marker-server \
    --server-mode balanced --server runpod-24gb --confirm-transit --server-cap-usd 0.40 \
    --spend-ledger ~/bench-runs/spend.jsonl --spend-cap-usd 4 --out out/report'
```

The window must hold the wait and the readiness floor of the attempt: with 0.40 USD it is 1,703 s, and 420 s of wait leave the 900 s of `ready_within` plus the margin of 60 s. The ledger counts such a window from the arm to the zero, because its command is not `mdf convert`; the bill counts only the worker running ([The ledger and the bill](#the-ledger-and-the-bill)). On Modal a container starts only with a request, so arming ahead gains nothing.

### Experimental levers of the start

Two levers of the start were measured on 22 and 23 November 2026 and not adopted: the memory snapshot of Modal (`snapshot`) and FlashBoot of RunPod (`flashboot`). They are not among the levers adopted (`DEFAULT_LEVERS`), and a session arms them only when its `--levers` names them. A session that arms one says so before its arm row, in a line of its own:

```json
{"adopted": false, "experimental_levers": ["flashboot"], "guide": {"flashboot": "docs/how-to/convert-documents.md#flashboot-of-runpod-experimental"}}
```

and again as `experimental_levers` in the `arm` row of the ledger and in the report of the window; nothing else changes. `--levers` replaces the levers adopted, so name in it the archives of the cache of compilation too ([The cache of compilation](#the-cache-of-compilation)), else vLLM compiles. With either lever the output was compared only on a one-page scan.

#### The memory snapshot of Modal (experimental)

Modal can keep a snapshot of the memory and of the GPU of a container and start the later containers of the same deploy from it. With the lever the session deploys `scripts/servers/modal_snap.py` in place of `modal_app.py`, with the same image, Volume, GPUs and limits. The first container of a deploy starts vLLM in its sleep mode on the loopback, warms it with a synthetic page made in the process, empties its caches and puts it to sleep, and only then does Modal take the snapshot: before the proxy starts and before any request, so no page of yours is ever in a snapshot. Every container, that one included, then wakes vLLM; a wake that fails or takes more than 120 s, or another type of GPU, falls back to a normal start, declared in the log (`mdf_fallback`). A snapshot serves one type of GPU, and a deploy changed in anything (code, image, levers, a fault) makes it obsolete. The snapshot of the GPU is an alpha feature of modal 1.6.3.

The lever runs only on Modal and only on an app named `*-snap` (`*-snap-fault` for the fault test), which takes exactly the levers of `providers.SNAPSHOT_LEVERS`: `snapshot` and the archives of `DEFAULT_LEVERS`. Another set, the lever on another app, `--allow-download`, `--disarmed` or `--cpu-activity` is refused before anything is read (`invalid_setting`). The app has a URL of its own: a first session with the command `true` prints it in its `armed` event, for `servers.json` (a `modal` server like `modal-l4`, here `modal-snap`) and for `--url`:

```bash
CC=compile-cache:29012f5b05f56b51698c99f6d5f573d3511ae06191946be983554a8b35a55b2b,compile-cache:6222ea36b0e660ccbd89b25386a30aab3dbaec14231bd36e7aae4f3b9751e7f4,compile-cache:cd31ce211155dae8d34230a9988f7fd82e855c54b94dc4e79607a7bc93608224
L=~/bench-runs/spend.jsonl
.venv-dev/bin/python -I -B scripts/servers/session.py run --provider modal --name mdf-surya-snap \
    --levers "snapshot,$CC" --env-file .env --spend-ledger "$L" --spend-cap-usd 15 --cap-usd 0.10 -- true
.venv-dev/bin/python -I -B scripts/servers/session.py run --provider modal --name mdf-surya-snap \
    --levers "snapshot,$CC" --url "https://<workspace>--mdf-surya-snap-serve-web.<region>.modal.run" \
    --env-file .env --spend-ledger "$L" --spend-cap-usd 15 --cap-usd 1.00 -- \
  .venv-dev/bin/mdf --env-file .env convert report.pdf --route marker-server --server-mode balanced \
    --server modal-snap --confirm-transit --server-cap-usd 1.00 --spend-ledger "$L" --spend-cap-usd 15 --out out/report
```

Measured on 23 November 2026, in three sessions of three or four cold starts each (NVIDIA A10 and A10G, vLLM 0.25.1, a one-page scan converted by `mdf convert` at each start):

- inside a session, a start from the snapshot was ready 8.0 to 10.0 s after its request (median 8.9 s; 6 restores of 6, no fallback), and `mdf convert` had its first answer after a median of 25.6 s. A normal start with the cache of compilation is ready in about 168 s on an A10;
- the first start of each deploy creates the snapshot and is slower than a normal start: ready 204 to 238 s after its request on an A10, 304 s on an A10G (each type of GPU creates its own);
- no reuse between sessions: a session deploys the app at its arm and stops it at its disarm, and after a stop and an identical deploy, less than a minute and 61 minutes later, Modal always created a new snapshot. Every `mdf convert` runs in a session of its own, so each one pays the creation;
- the output of the scan was identical byte for byte in the 10 conversions, from a snapshot or not.

The lever would serve an app kept up between several documents, which the product does not do: the first start would pay the creation (36 to 70 s more than a normal start on an A10), and the later ones would take about 9 s instead of about 168 s. The fault tests of a restored container (phases A and B with `--after-restore`, [the bench guide](bench.md#the-memory-snapshot-of-modal)) have not been run.

#### FlashBoot of RunPod (experimental)

With FlashBoot, RunPod does not stop a worker that stays idle past its idle timeout (60 s): it freezes its container at the level of the operating system, with its memory and the memory of its GPU, and resumes it for a later request instead of starting a new container. With the lever the arm turns FlashBoot on in its PATCH of the endpoint and reads it back; after the zero, also after a failed or interrupted command, the session turns it off with a PATCH of its own and reads it back (`flashboot_restored` in the `disarm` row; a failure is exit 4). After a session that died before that step (SIGKILL, a laptop that stopped), `session.py disarm` turns it off the same way after its zero, when the window it closes armed the lever or the endpoint still has FlashBoot on. At rest the endpoint keeps FlashBoot off: `verify-pins` wants it off and `align` turns it off. The watchdog tells a pause from a hang (a step of its own that overran its bound is a suspension, which it does not count against the container when no work was running), runs vLLM in its development mode with those routes closed at the edge, and at each passage to idle after work resets the prefix, multimodal and encoder caches of vLLM.

The lever runs only on RunPod. Name it beside the archive of the cache of compilation of the endpoint:

```bash
L=~/bench-runs/spend.jsonl
.venv-dev/bin/python -I -B scripts/servers/session.py run --provider runpod --name mdf-surya \
    --levers "compile-cache:<SHA-256 of the archive of the endpoint>,flashboot" --env-file .env \
    --spend-ledger "$L" --spend-cap-usd 4 --cap-usd 0.60 -- \
  .venv-dev/bin/mdf --env-file .env convert report.pdf --route marker-server --server-mode balanced \
    --server runpod-24gb --confirm-transit --server-cap-usd 0.60 --spend-ledger "$L" --spend-cap-usd 4 --out out/report
```

A resume can only happen inside one session, for a request that comes after the worker went idle: a command that sends documents with pauses of more than 60 s between them. The endpoint of the measures below, in CA-MTL-1, had the archive `compile-cache:6222ea36b0e660ccbd89b25386a30aab3dbaec14231bd36e7aae4f3b9751e7f4`.

Measured on 22 and 23 November 2026, in three sessions on `mdf-surya` and the fault test on `mdf-surya-fault` (RTX A5000 in CA-MTL-1, vLLM 0.25.1, a one-page scan):

- a resume was ready 13.8 to 22.8 s after its request (4 resumes, each 120 s after the request before), against 350 to 705 s for a cold start in that data center, where each session had a new host that loaded the image. The process thawed about 1 s after the request; the request reached it about 15 s later, a delay of RunPod's balancer;
- 2 resumes of 6 were missed: 660 s after the request before, RunPod had already stopped and removed the paused container (about 136 s into its pause); 120 s after, one request went to another worker that RunPod held ready, which started a container of its own;
- a paused worker is listed `IDLE` and is not billed: at closed hours the GPU seconds billed stayed within the seconds `RUNNING` in the log plus 30 s for each stop;
- no retention seen between sessions: no container of a session (its `boot_id`) came back in the next one, each session had another worker and another host, and each ran on a new version of the endpoint;
- a container frozen after a hang of vLLM resumed in the same state (the fault test `A-FB`: 503 `mdf_inert` from its proxy), and the endpoint reached the zero again;
- the output of the scan was identical byte for byte in the 4 conversions, cold or resumed.

**The risk.** A paused container keeps the state of its processes, the data of the documents it converted included (the page images and the text that vLLM and the proxy held in memory), frozen on a host of RunPod. RunPod describes the resume as starting "from exactly the state it was paused in" and releases paused containers after a timeout (its blog on FlashBoot, read on 21 November 2026); no page of RunPod documents that this state is wiped, nor when. The proxy resets the caches of vLLM at each passage to idle after work, which empties what vLLM keeps for reuse, not the memory of the processes; neither FlashBoot turned off, nor the disarm, nor the zero of the spend certifies that the state is gone. Keep sensitive documents on this machine (the `cpu` server or a `local` one), and arm the lever only with your explicit consent to this risk, for documents you accept to leave in a paused container.

### The ledger and the bill

The ledger (`--spend-ledger`, the `spend.jsonl` of a runs folder of the bench) records every session of a provider: the arm with its window, the reservations of the attempts, the disarm with the zero, and what each window counts. It never holds a key or document text.

```bash
.venv-dev/bin/mdf bench spend show --runs-dir ~/bench-runs
```

prints the total of each provider and of Datalab, every window with its amount and state (`open` while its session lives, then `clean`, `unclean` or `orphan`), and the windows over their bound (`over_bound`). The ledger counts upper bounds; the bill of the provider, read at an hour already closed, is the cost. A bound holds under two conditions: the caps on the side of each platform, set as in the preparation (on RunPod a small prepaid balance without automatic top-up, on Modal the spending limit and the usage budget), which no session can check; and a disarm whose zero comes within Z, which the session checks: a zero later than Z makes the window unclean, counted to the zero. A laptop that stops with a window armed leaves only the caps of the platforms and the stops of the watchdog in the container. On RunPod `runpod_endpoint.py --env-file .env read-history --name N --since ISO` prints the bill of the endpoint by hour; on Modal `uv tool run --from modal==1.6.3 modal billing report --for today -r h --json` prints it by app and hour. When an hour is closed, the figure billed for a window replaces its amount:

```bash
.venv-dev/bin/mdf bench spend reconcile --runs-dir ~/bench-runs --provider runpod --arm ARM_ID --manual-cents 4.07
.venv-dev/bin/mdf bench spend reconcile --runs-dir ~/bench-runs --provider modal --orphan --manual-cents 7.13
```

`--arm` (repeatable) names closed windows, `--activity` activities, `--orphan` the orphan windows that `session.py disarm` closed. A figure above the amounts of its windows is written and then stops the provider for good (`server_billed_over_bound`): a bill over the bound means that the guarantees of the session did not hold, and is a matter for you before any other session. The caps are never raised to let a session pass.

### A window that cannot be certified

A session counts its window within its reserve `A` only when it watched the whole window and saw the zero in time. Two cases make the close not certifiable, for a session of `session.py run` as for one of the archive:

- `unobserved_interval`: the wall clock moved more than 5 s away from the monotonic clock while the window was armed, because the laptop was suspended (Linux does not count a suspend in the monotonic clock) or its clock stepped. A drift that goes back counts at its largest. In that interval nothing was measured, also when it ended before the term `T`;
- `over_window`: the zero came after the limit of the window, `T + 60 s + Z` for a window not shortened (a slow disarm, a slow zero).

The window is then never cut at `A`: its `disarm` row has `orphan: true` and `uncertain` with the reason, the ledger counts the provider at its cap, and no new window opens on that provider (`session_orphaned`, also for `mdf convert` and the bench) until a reconciliation with the bill. `session.py run` writes the event `uncertain` in its report and exits with 4 (after 130 for a signal and 5 for the term, which come first: the row keeps the block). The archive shows the session as "GPU session not certified" with the reason, the hour of the zero, the provider at its cap and the command ([The GPU session of the archive](#the-gpu-session-of-the-archive)). Read on the bill of the provider the amount for the hours of the window (RunPod: `runpod_endpoint.py --env-file .env read-history --name N --since ISO`; Modal: `uv tool run --from modal==1.6.3 modal billing report --for today -r h --json`), when those hours are closed, then:

```bash
.venv-dev/bin/mdf bench spend reconcile --runs-dir RUNS --provider runpod --orphan --arm ARM_ID --manual-cents N
```

`RUNS` is the folder of the ledger (for the archive `MDF_GPU_LEDGER_DIR`). The figure replaces the cap and frees the provider. To keep a session from being suspended, run its commands under `systemd-inhibit --what=sleep:idle:handle-lid-switch`.

### Limits of the route

- In `balanced` mode a table cell written on several lines can lose part of its text or have it moved to another cell, because Marker rebuilds the tables on this machine from the lines the server reads; seen on the table of a scientific article with the CPU server and with the GPU servers. `accurate` reads the whole page; its result on such tables has not been measured.
- On a GPU server two conversions of the same document can differ in a few places, also on the same GPU with the same settings (a letter, the case of a word, the part of a multi-line cell that is kept): compare two outputs by their content, not by their bytes. Seen in `balanced` mode; `accurate` has not been measured for this.
- Modal chooses the GPU (an L4 when it has one, else an A10 or an A10G), and an archive of the cache of compilation serves only its own GPU.
- RunPod may list extra workers that never run and are never billed; a host that has not loaded the image takes 9 to 12 minutes to be ready.
- The local CPU server reads a page in minutes.
- The quality of the route is not certified, like every route: `engine.json` states its limits in `limits`, and the summary repeats them.

## Convert in the archive of the product

The archive keeps every source, every attempt and every result under `MDF_DATA_ROOT` (default `./mdf-data`). It runs as two processes on the same machine, which load the same settings (`--env-file`, see [`.env.example`](../../.env.example)):

```bash
.venv-dev/bin/mdf --env-file mdf.env serve      # API and Web UI on http://127.0.0.1:23465
.venv-dev/bin/mdf --env-file mdf.env worker     # the conversions, one at a time
```

Open `http://127.0.0.1:23465` in a browser on the same machine, choose files under "Choose one or more files" and press "Convert"; a conversion shows its progress, then the extracted Markdown, the images, "Download source" and "Download Markdown and assets (ZIP)". From the command line, `mdf submit FILE… [--wait] [--route R] [--ocr force] [--lang L] [--datalab-mode M] [--confirm-transit]` sends files to the same archive, and `mdf jobs`, `mdf show ID`, `mdf retry ID [--confirm-transit]`, `mdf cancel ID`, `mdf download ID --output FILE` follow them. A refusal prints its code next to the text (`{"error": …, "code": "transit_not_confirmed"}`) and exits 1.

The settings file of the product holds no key: `mdf serve` and `mdf worker` read from it only the lines of the `MDF_` settings. A file given to `mdf worker --env-file` must be a regular file of yours with mode 0600, not a link (`server_key_unsafe` otherwise, before anything reads it), because it may also hold the keys of the GPU sessions; the worker takes the keys of the rented GPUs (`RUNPOD_API_KEY`, the Modal proxy token, the `key_env` of `servers.json`) out of its environment when it starts.

### Conversion presets

A conversion preset says how a document is converted: route, mode of the GPU route, GPU preset, OCR, mode of Datalab. The archive comes with four, provisional until their values are measured:

| Preset | What it does |
| --- | --- |
| `local-fast` | the route of the matrix for each format on this computer: Marker `fast` for pages and images |
| `quality-now` | pages and images on a GPU of Modal (`marker-server`, `balanced`, `modal-base`) |
| `batch` | pages and images on a GPU of RunPod (`marker-server`, `balanced`, `runpod-serverless-volume-dc`) |
| `datalab` | pages and images with Datalab, `balanced` |

`quality-now` and `batch` run on a GPU preset of the instance ([The GPU session of the archive](#the-gpu-session-of-the-archive)): when the instance does not enable it they are listed and unavailable with their code (`gpu_preset_not_enabled`, `server_cap` with a cap of 0, `session_orphaned` while a window of the provider waits for the bill). You add presets of your own with the same fields and choose your default:

```bash
.venv-dev/bin/mdf --env-file mdf.env presets list
.venv-dev/bin/mdf --env-file mdf.env presets add scans --route marker --ocr force
.venv-dev/bin/mdf --env-file mdf.env presets default scans      # or: presets default --clear
.venv-dev/bin/mdf --env-file mdf.env submit page.png --preset scans --lang it
.venv-dev/bin/mdf --env-file mdf.env presets remove scans
```

The names of the presets of the archive are reserved (`preset_reserved`); a name is lowercase letters, digits and hyphens (`preset_invalid` otherwise, as for options outside their form), once per user (`preset_exists`). With `--preset` only `--lang` and `--confirm-transit` go beside it; the job records the name of the preset in its options (`preset`). The API has the same operations: `GET /api/v1/presets` (the presets, the default and the GPU presets with whether this instance enables them), `POST /api/v1/presets` with `{"name": …, "options": {…}}`, `PUT` and `DELETE /api/v1/presets/NAME`, `PUT /api/v1/settings/default-preset` with `{"name": …}` or `null`, and `POST /api/v1/runs?preset=NAME`, with only `lang` and `transit_confirmed` beside it.

The GPU presets an instance offers are its settings (in `.env.example`): `MDF_GPU_PRESETS` names them (only supported presets on a network volume), `MDF_SERVERS` is the `servers.json` of the instance, `MDF_GPU_LEDGER_DIR` the folder of its spend ledger, outside the archive, `MDF_GPU_CAP_USD_MODAL`, `MDF_GPU_CAP_USD_RUNPOD` and `MDF_GPU_SESSION_CAP_USD` the caps. None of them is part of the identity of the archive.

### The GPU session of the archive

A job on a GPU preset (`quality-now`, `batch`, or `route=marker-server&server_mode=balanced&gpu_preset=NAME&transit_confirmed=true`) runs on a session of that preset, which the worker opens at the first such job, keeps while the queue has jobs of the preset, and closes 60 s after the last one (`idle_s` of the preset). One session at a time: its jobs go first, a job of another preset closes it, and a local job queued after it closes it at once. Only the worker talks to the provider, each call in a child process with its own limit; the API writes requests and reads states.

What the instance needs:

- `MDF_GPU_PRESETS` with the preset, its caps above 0, and an endpoint of `MDF_SERVERS` that names it ([GPU presets](#gpu-presets));
- the keys of the endpoint by name in the env file of the worker (`mdf --env-file worker.env worker`, a file of yours with mode 0600: `RUNPOD_API_KEY`, or `MODAL_PROXY_TOKEN_ID` and `MODAL_PROXY_TOKEN_SECRET` with the token of the SDK in `~/.modal.toml`), never in the settings of `mdf serve`;
- for Modal, `uv` on the `PATH` of the worker (the CLI of the SDK).

Before anything is armed the worker opens a session only when it fits: no orphan window of the provider in the ledger, no window of another session (`provider_not_idle`), the endpoint at rest, and on RunPod its pins and, since this version, those of the preset: the volume of the preset in its data center, of its class and size (`endpoint_out_of_pins`). Only the session of the archive passes the preset to these checks (`pin_checks`, `RunPod.arm`); a session of `session.py run` checks the pins of MFL033 only. The reserve of the session `A` is the smaller of `MDF_GPU_SESSION_CAP_USD` and what the cap of the provider leaves, its window `W = A / rate - 60 s - Z - tail`, and it must hold an attempt of `MDF_MAX_PAGES` pages (one page to arm ahead): that is the **minimum reserve** of a session, 45.23 cents on Modal and 28.76 cents on RunPod with the 50 pages of the default, a reserve and never a charge. A larger `MDF_MAX_PAGES` raises it (98.19 cents on Modal with 200 pages); under it the job fails with `server_cap` before anything is armed.

The Web UI shows, before you send a job with a GPU preset, the provider, its region and GPU, where the page images go, the reserve and window of a session now, its minimum reserve, about how much a start and a page cost and the measures of the preset with their date, hardware and source (provisional values, quality not verified); "Convert" stays disabled until the transit is confirmed. The same estimate is `GET /api/v1/gpu/estimate?preset=NAME` and `mdf gpu estimate NAME`. A job then shows the reserve and the measured limit of its attempt: an upper bound, not a bill.

To arm ahead (RunPod only: see [Arm ahead](#arm-ahead)), confirm the transit and press "Arm ahead", or `mdf gpu arm runpod-serverless-volume-dc --confirm-transit`, or `POST /api/v1/gpu/arm` with `{"gpu_preset": …, "transit_confirmed": true}`: the worker arms between two jobs (a request older than 10 minutes is refused), and the session closes by itself when its window can no longer hold a page (264 s after the arm with a reserve of 0.30 USD). "Disarm" in the section "GPU session" (or `mdf gpu disarm NAME`, `POST /api/v1/gpu/disarm` with `{"gpu_preset": …}`) stops the session at once: the conversions of that preset in progress and in the queue are cancelled. `GET /api/v1/gpu/session` and `mdf gpu session` give the state of the session.

The ledger of the instance is `MDF_GPU_LEDGER_DIR/spend.jsonl`; give the same file to `mdf convert --spend-ledger` and the same folder to `mdf bench run --runs-dir` so that they and the archive never arm the same endpoint at once (`provider_not_idle` in both directions). If Modal has no GPU for 300 s, the job fails with `modal_no_capacity` ("Modal had no GPU: retry later or choose the preset Batch (RunPod)"); the provider is never changed for you. When a worker stops with a session armed, its next start disarms the endpoint before taking any job and holds the window as an orphan at the cap of the provider until you reconcile it with the bill ([A window that cannot be certified](#a-window-that-cannot-be-certified)); the section "GPU session" shows the command. If it cannot close such a window (its ledger missing, a key missing, no zero), the worker stops before any job with `gpu_unrecoverable` and the commands that close it by hand. A session that cannot be certified is shown as "GPU session not certified" with the same command.

### Formats and routes

The archive accepts every format of the conversion matrix, recognized by the extension of the file name and confirmed by the signature of the whole file: a `.docx` that is not a Word package is refused with `format_mismatch`, an extension outside the matrix with `format_unsupported`, a damaged package with `input_unreadable`. The source is kept with its extension (`sources/<id>.docx`), and the ZIP and "Download source" give it back as `source/document.<ext>` with its media type.

The route is the default of the format unless you choose one of its alternatives (the table in [Choose a route](#choose-a-route)); `marker-server` runs in the archive only on a GPU preset of the instance, usually through a conversion preset, and without one it is refused with `route_not_in_product` (a local or own server is for `mdf convert`). The Web UI preselects "Default of the format (recommended)" and lists only the routes every chosen file has. The API takes the options as query parameters of `POST /api/v1/runs` (`route`, `ocr`, `lang`, `datalab_mode`, `transit_confirmed`, and on the GPU route `server_mode` and `gpu_preset`); a retry of a GPU job confirms the transit again; the API and the worker both validate them, and a refusal carries its code next to the text (for example `{"error": "This format has no such route", "code": "route_invalid"}`). `GET /api/v1/session` lists the formats with their routes and `GET /api/v1/readiness` the availability of each route as the worker last checked it. A route that cannot run on this machine (for example Marker without its weights, `weights_missing`) fails its job with `route_unavailable` and the next jobs go on.

The detail of a completed conversion lists its problems with their page under "Problems" and the limits of its route under "Limits of this conversion" (the limit of Marker on vector and scanned figures is one of them). A run of a previous version keeps its options: `force_ocr` counts as `--ocr force`.

### Datalab in the service

Datalab appears in the Web UI only when the worker has the key and the spending cap of each conversion is above 0:

- the cap is `MDF_DATALAB_CAP_USD` (for example `0.05`) in the settings of both processes; the default 0 sends nothing;
- the key is `DATALAB_API_KEY` in the environment of the worker process only, never in the settings file of the product. With the key alone in a file of yours with mode 0600 (here `datalab.env`), start the worker so that it is not printed or kept in the shell:

  ```bash
  ( set -a; . ./datalab.env; set +a; exec .venv-dev/bin/mdf --env-file mdf.env worker )
  ```

  The worker takes the key out of its environment when it starts, so no child process inherits it; it is never written to the archive, the manifests or the logs.

Before sending, the Web UI shows what happens to the document under "Datalab: the document leaves this computer" (it is uploaded to the Datalab API, its result is deleted there afterwards, and whether Datalab keeps it or uses it for training depends on the settings of the Datalab account, not on the app) and the button stays disabled until "I confirm that these documents are sent to Datalab" is checked. The API needs `route=datalab&transit_confirmed=true`; without the confirmation the request is refused with `transit_not_confirmed`. A retry sends the document again, so it needs the confirmation again ("Send it to Datalab again"; `mdf retry ID --confirm-transit`; `POST /api/v1/runs/ID/retry?transit_confirmed=true`); without it the retry is refused with `transit_not_confirmed` and no new run is queued. The detail of the run shows that the document left the computer, whether its result was deleted on Datalab and the estimated and billed cost.

### The archive identity and the return to a previous version

The archive records the identity of the settings it depends on (identity schema 3, profile `engine-v1`): the port, the model root, the time limits, the Datalab cap and the settings of the GPUs are not part of it, the quotas and the limits of a conversion are (`mdf configure` changes the storage quota). An archive of a previous version (identity schema 1 or 2) is upgraded by its first opening, which needs every API, worker and CLI of every version stopped; the upgrade is recorded in the `history` of `identity.json`. If it was created with a former default port, open it once with `MDF_PORT=23460` (or `8765` before 2026-11-07); with the former default model root (`./mdf-models` of the directory it was used from), once with `MDF_MODEL_ROOT` set to the absolute path of that `mdf-models` directory, from any directory, or with `MDF_MODEL_ROOT=./mdf-models` from the directory it was used from; the mismatch message names them.

Its database is at schema 5 (4: the conversion presets of the users and their default; 5: the GPU sessions, their requests and the GPU facts of each run), which an archive of schema 3 or 4 reaches at its first opening, with the same identity. A binary of an earlier schema refuses it (`Unsupported database schema`): to use the archive again with the binary of schema 4 or 3, stop every process and add `--to-schema 4` or `--to-schema 3` to the command below. Those returns leave `identity.json` as it is. The return under 5 is refused, without changes, while a GPU session is not closed (`uncertain`, `orphan` or `stopped`: reconcile it first), a request to arm or disarm is pending, or a run has GPU facts or ran on the route `marker-server`; the return under 4 while a user has a conversion preset or a default, or a run was submitted with a preset (its options hold `preset`).

To use the archive again with the version of schema 2, stop every process and run:

```bash
.venv-dev/bin/mdf --env-file mdf.env downgrade-storage --expected-config "$(python3 -c 'import json; print(json.load(open("mdf-data/identity.json"))["config"])')"
```

The return is refused, without changes, unless the archive was upgraded from schema 2 (or 1) with the same settings, no run is running, every source is a PDF, every run has the options of that version and the owner `local`, plus the conditions of the return to schema 3: once a file of another format or a run with the options of schema 3 is in the archive, it can return to schema 3 at most. The checks of every schema crossed run before anything changes.

## Limits

- Marker 2.2.0 `fast` with indra-ocr v1.1.0 neither extracts nor detects vector figures (also the vector panels of a composite figure) nor the figures of scanned pages: they are missing without a problem of the document; `engine.json` states it in `limits` and the summary repeats it.
- A short text file in a legacy 8-bit encoding can be detected with another encoding: check `text.encoding` in `engine.json`.
- The `marker-server` route has its own limits, in `limits` of `engine.json` and in the summary: see [Limits of the route](#limits-of-the-route).
- The quality of a conversion is not verified: a complete status means that the contract and its checks hold, not that the content is faithful.

[Index](README.md).
