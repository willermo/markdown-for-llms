# Web UI — User path and progressive development

Status: initial requirements and implementation proposal. The first increment I1 provides a minimal inert text UI ([ADR 0009](../decisions/0009-first-usable-product.md)); the rest of this page describes the target experience. Reference decision: [ADR 0002](../decisions/0002-fastapi-web-ui.md).

## Expected experience

The Web UI is a central part of the product. It must be usable in a local browser and distributed together with the backend, avoiding the need for a different GUI for each operating system. Swagger remains available to explore the API (target experience; I1 exposes only the local OpenAPI JSON, without Swagger or CDN).

| View | Actions and information |
| --- | --- |
| New conversion | Upload files or bundles, choose profile and languages, see whether data will be processed locally or by a remote provider |
| Queue and activity | See the current phase, errors, cancellation and repetition; show indeterminate progress if the engine does not provide a reliable percentage |
| Archive | Search documents and conversions, compare versions, reopen results and download bundles |
| Review | Source and result side by side, problems per block, OCR proposal with before/after and reason, acceptance or rejection |
| Result | Preview of formulas, images and tables; Markdown text; download of the document and of the assets |

For PDF documents the source panel shows the page and the region associated with the block when available. For native formats without coordinates, use sections and logical identifiers; the UI must not invent a position in the page. For long documents load pages and previews progressively.

## Understandable states

Distinguish execution state, availability of the artifacts and review state. A finished job can produce a result to be checked. A conversion error, a missing asset and a suggested correction are different conditions.

Accepted corrections create a revision and keep the original extraction. The download makes clear which revision is exported. Do not use a "verified" label based only on the confidence declared by the LLM.

## Frontend choice

React with TypeScript is a proposal to compare with a simpler frontend served by the backend. The comparison must include PDF viewer, page/block link, rendering of mathematics, review of differences, accessibility and maintenance. For I1 the choice was made in [ADR 0009](../decisions/0009-first-usable-product.md): HTML, CSS and JavaScript without SPA framework or CDN; React with TypeScript remains an option to evaluate when comparison and review justify its cost, without introducing frontend dependencies or a component library out of inertia.

## Increments and verification

1. **Phase 3:** upload, queue, archive and download really connected to the backend; preview of a simple format, persistence also after a restart.
2. **Phase 4:** display of the scientific sources and of the conversion problems.
3. **Phase 5:** comparison per block, OCR/AI corrections and tracked revisions.

Verify the path also with failed jobs, browser refresh, long documents, formulas and missing assets. Provide keyboard use and readable messages; keep credentials on the backend and sanitize the rendered contents. Network access and multi-user management depend on the scope still to be defined, while the first proposed distribution is local.
