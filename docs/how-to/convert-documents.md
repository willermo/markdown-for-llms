# Convert a document with `mdf convert`

For Linux users and developers. `mdf convert` converts one file directly with the conversion engine (`markdown_for_llms/engine/`), without the archive, the API or the worker: the output directory receives the complete Markdown document, its images and a machine-readable result. To send files to the archive of the product instead, use `mdf submit` ([reference](../reference/local-product.md)).

## Before you start

- The development venv of this repository (see [Prepare and update the uv environment](uv-environment.md)), with the package installed.
- Docker Engine on Linux with a local socket, and the toolchain image built once from a clean checkout. `mdf prepare --toolchain` is the same command as `mdf bench prepare --toolchain` ([Prepare the reference bench](bench.md)):

  ```bash
  .venv-dev/bin/mdf prepare --toolchain
  ```

- Bubblewrap at `/usr/bin/bwrap`: the audit of OOXML, OpenDocument and EPUB packages runs in a child without network.

Nothing is downloaded during a conversion, and no document leaves the machine.

## Convert a file

```bash
.venv-dev/bin/mdf convert report.docx --out out/report
```

`--out` must be a new or empty directory. On success it holds:

| Path | Content |
| --- | --- |
| `document.md` | the document in Markdown (CommonMark with the pandoc extensions of `commonmark_x`), UTF-8 |
| `assets/` | the images the document references, named `asset-NNNN.<ext>` |
| `diagnostics/pre-filter.md` | the extraction as pandoc produced it, before the corrections of the engine |
| `engine.json` | the result: source and its SHA-256, route, tool steps, versions, inventory of every file with bytes and SHA-256, corrections and problems |

stdout carries one JSON summary (status, closed error code, route, number of files, problems); it never contains document text. On a terminal, stderr shows the steps. The exit status is 0 for a complete conversion, also when it has problems, 1 for a failure, 2 when the command is refused before converting (for example an output directory that is not empty).

A failed conversion leaves only `engine.json` (with `status: failed` and the closed code, for example `format_mismatch`) and `diagnostics/` with bounded logs; never a partial `document.md`.

## Read the problems

A complete conversion can carry problems, each with a severity:

- `incomplete`: content of the source is missing from the output. `media_lost` (an image the content uses that is not in the document, for example a picture inside an OpenDocument text box, an EPUB figure drawn by CSS, or an image the Markdown writer leaves out, such as an image nested in the alternative text of another inside a table written as HTML, also when its bytes are inline in the source), `media_missing` (an image whose source is not part of the document, for example an absolute path of a saved web page), `object_not_converted` (a chart, an OLE object or an inline SVG drawing, also an OpenDocument chart without a preview image in the document).
- `warning`: nothing of the document's content is missing. `media_furniture` (an image of a slide layout, a header or a footer), `media_unused` (a media file no part references), `media_external` (a remote image, left as a link and never fetched), `page_blank` (a DjVu page without text).

Read them in `engine.json`: `problem_counts` (per severity) and `problem_codes` (per code) count all of them, `problems` lists at most 2,000, the `incomplete` ones first; the stdout summary carries the same counts. Links to targets outside the document (site paths, `javascript:`, files that are not part of the output, destinations that do not decode cleanly) are unlinked with their text kept, and active HTML is removed: these are recorded under `corrections`, not as problems.

## Choose a route

Each format has a default route; `--route` imposes one of its alternatives.

| Format | Default | Alternatives | In this version |
| --- | --- | --- | --- |
| docx, xlsx, pptx, epub, html | `pandoc` | `marker` | pandoc available; Marker not yet |
| odt, rtf | `pandoc` | — | available |
| doc, ods | `pandoc`, after LibreOffice converts them to docx and xlsx | — | available |
| txt | `text` (encoding detected by charset-normalizer, text kept verbatim) | — | available |
| djvu | `marker` | `djvutxt`, `datalab` | `djvutxt` available: the text layer, page by page |
| pdf, png, jpeg, tiff | `marker` | `datalab` | not yet: `route_unavailable` |

```bash
.venv-dev/bin/mdf convert book.djvu --route djvutxt --max-pages 300 --out out/book
```

`--max-pages` (default `MDF_MAX_PAGES` of the product settings, 50) bounds the pages of a DjVu file. `--lang` is recorded in `engine.json`; no converter of this version uses it. `--ocr force` is refused with `route_option_unsupported` by every route of this version. The memory cap of each tool (`MDF_MEMORY_MIB`, at most 2,048 MiB for the package audit), the output cap (`MDF_OUTPUT_MIB`, `engine.json` included, also the cap of the private working directory of each tool) and the total deadline (`MDF_TIMEOUT_SECONDS`, 0 for none) come from the product settings; `--env-file` of `mdf` selects them.

## Limits

- The pages family (PDF and images) and the Marker and Datalab routes are not available in this version of the engine.
- A short text file in a legacy 8-bit encoding can be detected with another encoding: check `text.encoding` in `engine.json`.
- The quality of a conversion is not verified: a complete status means that the contract and its checks hold, not that the content is faithful.

[Index](README.md).
