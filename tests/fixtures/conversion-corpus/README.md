# Corpus sintetico di conversione

Dieci canonici originali con gold manuali per documento. `manifest.json` identifica input, gold, asset, nove pagine PDF e dodici render fissi. EPUB è reflowable; TeX ha un main e due include e non è compilato. Le derivazioni raster sono sotto `sources/derivations`.

Il gold rimane PROVISIONAL finché M e la verifica umana V non accettano esattamente gli hash correnti. I render e la presenza del file non certificano una conversione. `controls/` contiene controlli autonomi del misuratore, non output di motori. Licenze e font in `NOTICE`.

Revisione 5: `header_rows` e `header_policy` sono dichiarazioni semantiche indipendenti dal motore. DOCX/RTF/TeX sono headerless; EPUB ha celle `th`; i controlli pipe hanno header reale. PDF e raster usano una policy autoriale headerless: nessuna semantica encoded viene inferita dai glifi. La riga pipe vuota richiesta dalla sintassi è distinta dal corpo. Sono state completate le 11 caption di tabella nelle nove trascrizioni interessate; canonici, asset e render non sono stati rigenerati.

L'eccezione `.gitignore` riguarda esclusivamente i PDF sintetici sotto questa directory: tutti i venti PDF canonici, fonti e derivati sono versionabili. Le receipt V della revisione precedente restano storiche; le nuove qualificazioni richiedono M e una receipt derivata di invarianza del contenuto già approvato, senza una nuova conferma umana.

## Path-only revision (2026-10-18)

Provenance `path` fields (`manifest.json` `provenance`, `controls/manifest.json` `source`, gold `metadata_revision.previous`) and the `%%Image` line of the three raster scripts in `sources/derivations/` record repository-relative paths instead of absolute ones. Each provenance `path` still pairs with its unchanged `bytes` and `sha256`; content, locators and denominators are unchanged. The 13 gold files therefore have a new sha256, and both manifests carry the new `gold_sha256`. Earlier M receipts and derived qualifications refer to the previous hashes; nothing VERIFIED is copied to the new ones.

| Gold | Previous sha256 | New sha256 |
| --- | --- | --- |
| `gold/C01-txt-it/annotations.json` | `8bbd7d35e921cabe94add5471f8915fde578a83c70c40bdb9eb93520ae079e99` | `893bae93eab1e53c1bfba6257d2bc04920133170136fa895a69c0b8e642641f2` |
| `gold/C02-docx-it/annotations.json` | `ec28e4f65469471890fc273d1dc404841654ed928683a23a6ca593ad2639e393` | `f1f5e3a00c236eee39aff5292ebc2e435fdb82c0232f3b9db9fa926d16ee3e58` |
| `gold/C03-epub-en/annotations.json` | `00023055031057b583de22ef96a75488563989db517e9d41372847c23038fdad` | `07e68c88a5be78c4633c5366eee29fc7d2a5a1112800a49c5c6f51510f981131` |
| `gold/C04-rtf-it/annotations.json` | `7f0255017fa3f0ff17bf82462713c327faa154d073ff60a611dab3e45f823bfd` | `0561d76ec8970c428996941c72b1d9fd9238a1c5dfe084c2b269d91261726386` |
| `gold/C05-tex-en/annotations.json` | `ab3179d593d2c6acfa27881cc3affcec586af17863edc00c83497508a9f88de6` | `daa6743e1c81804a5a9aa05169f77a6de5b1f4555f7a545bb2d8b04072ad494c` |
| `gold/C06-pdf-text-it/annotations.json` | `b287404c1eec751847589dc2b3e0d2ce4ad6a7d85608c8ff4c8afe8fc30f6e62` | `673c4896537933d0a4f4544356351491f3d14477b30ff2b6435e6b36d93c796e` |
| `gold/C07-pdf-columns-en/annotations.json` | `2c2460e0008d3ade8392a6a23bf1877fe787a55e5dfb062b1b4eb2c696d3b15a` | `39eac9463eb0f475041d3d4d046b7a2ee17102cbf9fb114f3c294e3ab371e174` |
| `gold/C08-scan-it/annotations.json` | `80340cad1e68c767772a6de423cf7cfcaa824808b2d53c98d043fab33e19c85b` | `431011201af324f9d9e3f82fe535ac4f7d6fe9f66011a7ca9d6473b367a6e2d4` |
| `gold/C09-scan-en/annotations.json` | `1df7ccaa55a1052a56efde43580c1560a0075ea9fe743e484a68f6c1f1b70766` | `8c46b67e6736d6546bbdb5c81a1abeb84184ff02aa31629e159962e5a2657ea0` |
| `gold/C10-pdf-mixed-it-en/annotations.json` | `ab7e4b2e94422d3e656075ca8b1e39e5e54f767dd49fafa9e028e672912015bb` | `23713fd166d476c8aaabcd001e99af0538f6daa803c125574945e8f45670f460` |
| `controls/K01-full/annotations.json` | `80cc89e22be0094271d63f7d08fa9dcace08516d309f16f76943259a81c76961` | `83f3d16ee40333c0d4c674540816682b945c5582255ccb8e2c53fcc5a18ab39f` |
| `controls/K02-three-sections/annotations.json` | `0aa2d1aa05a04f19ed04ed164961972481e84392152269411dfaf3c00281a75e` | `c6b56626b1eba5a56026e523ba422f6356ef0549fa5f8c5c948d367443e1c9b5` |
| `controls/K03-repeated/annotations.json` | `704bfecd4f5d197fa2a30d7a679336cfc6b6d9460ab8d9d4fb314949915496fb` | `fc55f127b2669ba3071b11782e55c6fb6b505beeeae5b09b2888c225a02a15b9` |
