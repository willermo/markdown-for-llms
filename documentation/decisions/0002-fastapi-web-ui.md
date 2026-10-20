# ADR 0002 — FastAPI, separate worker and Web UI

- Date: 2026-10-02
- Decision status: accepted
- Implementation status: I1 r004 GO, integrated into dev at f8bf44c and push verified on 2026-10-17; static UI according to ADR 0009, advanced features later
- Origin: explicit preference for FastAPI and the user's request for a GUI/Web UI

## Context and decision

Adopt FastAPI for the API of the new application. The application services must also be callable from the CLI. Run OCR and long conversions in a worker separate from the process that answers the HTTP requests, with persistent job state. Queue technology and the precise job schema remain to be designed (at the time of the decision; for I1, ADR 0009 adopts a SQLite queue with claim/lease, see the refinement below).

Provide a Web UI usable in the browser on Linux, macOS and Windows: upload, progress, archive, preview of Markdown and assets, comparison with the source, review of the OCR proposals and download. Introduce the minimal path in the first version that integrates API, worker and storage; complete the review in the later phases.

FastAPI generates API documentation with OpenAPI and Swagger UI; these capabilities support developers. The interface to convert and review documents requires a dedicated frontend. [FastAPI source](https://fastapi.tiangolo.com/features/).

## Alternatives and consequences

A desktop GUI would impose additional distributions and updates. A Web UI served together with the application meets the local and cross-platform requirement. React with TypeScript is a candidate, not a dependency already approved or installed. The choice will be justified by preview, comparison, maintenance and development cost (for I1, ADR 0009 chose HTML, CSS and JavaScript without a framework; React with TypeScript remains an option when comparison and review justify its cost).

The speed of the API does not determine that of OCR: measure HTTP response, queue wait and inference separately. The AI libraries stay behind adapters, without tying the conversion logic to the HTTP framework.

Details: [Web UI design](../architecture/web-ui.md), [roadmap](../roadmap.md).

## Refinement of 2026-10-17 — First increment

[ADR 0009](0009-first-usable-product.md) adopts HTML/CSS/JavaScript served by FastAPI, with inert text and a raster gallery, and a SQLite queue with a separate worker. CLI on the same services. First increment delivered and GO r004 after two independent reviews; integrated into dev at f8bf44c, push verified. Advanced comparison and review remain later; initial scope proven on Linux CPU/loopback.
