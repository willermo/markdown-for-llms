# Convert a document with `mdf convert`

For Linux users and developers. `mdf convert` converts one file directly with the conversion engine (`markdown_for_llms/engine/`), without the archive, the API or the worker: the output directory receives the complete Markdown document, its images and a machine-readable result. To send files to the archive of the product instead, use `mdf submit` ([reference](../reference/local-product.md)).

## Before you start

- The development venv of this repository (see [Prepare and update the uv environment](uv-environment.md)), with the package installed.
- Docker Engine on Linux with a local socket, and the toolchain image built once from a clean checkout. `mdf prepare --toolchain` is the same command as `mdf bench prepare --toolchain` ([Prepare the reference bench](bench.md)):

  ```bash
  .venv-dev/bin/mdf prepare --toolchain
  ```

- Bubblewrap at `/usr/bin/bwrap`: the audit of OOXML, OpenDocument and EPUB packages and the preparation of PDFs run in a child without network.
- For the pages family (PDF, images, DjVu) on the default route, Marker 2.2.0 on the CPU: the development venv has it (extra `engine`); its 18 weight files (indra-ocr v1.1.0 and the GoNoto font, about 111 MB) prepared once, and a systemd user manager (`systemd-run --user --scope`), which caps the memory of the Marker child:

  ```bash
  .venv-dev/bin/mdf prepare --weights --model-root ~/.cache/markdown-for-llms/models
  ```

  `mdf prepare --weights` is the only command of the engine that uses the network: it downloads each file by commit, verifies bytes and SHA-256, retries a few times, resumes from the files already verified and never overwrites one that does not verify. Every conversion verifies the 18 files again before it starts the child (`weights_missing`, `weights_mismatch`). Pass the same root to every conversion, `mdf --model-root ~/.cache/markdown-for-llms/models convert …` (or `MDF_MODEL_ROOT`; its default is still `./mdf-models`); the worker of the product uses `MDF_MODEL_ROOT` too.

Nothing is downloaded during a conversion, and no document leaves the machine except on the Datalab route, which you choose and confirm (below).

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
| docx, xlsx, pptx, epub, html | `pandoc` | `marker` | available (`marker`: Marker's own provider makes a PDF first) |
| odt, rtf | `pandoc` | — | available |
| doc, ods | `pandoc`, after LibreOffice converts them to docx and xlsx | — | available |
| txt | `text` (encoding detected by charset-normalizer, text kept verbatim) | — | available |
| djvu | `marker` (after `ddjvu` makes a PDF of images) | `djvutxt`, `datalab` | available; `djvutxt` keeps the text layer, page by page |
| pdf, png, jpeg, tiff | `marker` (an image becomes a PDF page, a TIFF one page per frame) | `datalab` | available |

```bash
.venv-dev/bin/mdf convert book.djvu --route djvutxt --max-pages 300 --out out/book
```

`--max-pages` (default `MDF_MAX_PAGES` of the product settings, 50) bounds the pages of the pages family (`page_limit`). `--lang` is recorded in `engine.json`; no converter of this version uses it. `--ocr force` reads every page with OCR on the Marker route and is refused with `route_option_unsupported` by the others. The memory cap of each tool (`MDF_MEMORY_MIB`, at most 2,048 MiB for the package audit), the output cap (`MDF_OUTPUT_MIB`, `engine.json` included, also the cap of the private working directory of each tool) and the total deadline (`MDF_TIMEOUT_SECONDS`, 0 for none) come from the product settings; `--env-file` of `mdf` selects them.

## The Marker route

Marker 2.2.0 runs in `fast` mode on the CPU, in a Bubblewrap child without network, the weights read-only, under a systemd scope whose memory cap is `MDF_MEMORY_MIB`; a child that writes no progress event and uses no CPU time for `MDF_STALL_SECONDS` is stopped (`stalled`). The PDF is classified first, page by page (`pages` in `engine.json`): `native` (visible text), `ocr_layer` (an image over most of the page with only invisible text, including the glyphless layers of OCR programs) or `image`.

- Only native pages: Marker reads their text; Indra, its reader, still runs OCR where the text layer is broken (for example a broken font map or text without ink) and says so per page: `indra_source` `hybrid` (part of the page) or `ocr` (the whole page), with the reason in `note`.
- No native page, or `--ocr force`: every page is read with OCR.
- Native pages with OCR-layer pages: the invisible text of those pages is removed and the copy is checked (the same rendering, no layer left); if the copy renders differently the original goes with OCR on every page (`ocr_forced_document`).

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

## Limits

- Marker 2.2.0 `fast` with indra-ocr v1.1.0 neither extracts nor detects vector figures (also the vector panels of a composite figure) nor the figures of scanned pages: they are missing without a problem of the document; `engine.json` states it in `limits` and the summary repeats it.
- A short text file in a legacy 8-bit encoding can be detected with another encoding: check `text.encoding` in `engine.json`.
- The quality of a conversion is not verified: a complete status means that the contract and its checks hold, not that the content is faithful.

[Index](README.md).
