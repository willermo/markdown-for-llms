# Corpus sintetico di conversione

Dieci canonici originali con gold manuali per documento. `manifest.json` identifica input, gold, asset, nove pagine PDF e dodici render fissi. EPUB è reflowable; TeX ha un main e due include e non è compilato. Le derivazioni raster sono sotto `sources/derivations`.

Il gold rimane PROVISIONAL finché M e la verifica umana V non accettano esattamente gli hash correnti. I render e la presenza del file non certificano una conversione. `controls/` contiene controlli autonomi del misuratore, non output di motori. Licenze e font in `NOTICE`.

Revisione 5: `header_rows` e `header_policy` sono dichiarazioni semantiche indipendenti dal motore. DOCX/RTF/TeX sono headerless; EPUB ha celle `th`; i controlli pipe hanno header reale. PDF e raster usano una policy autoriale headerless: nessuna semantica encoded viene inferita dai glifi. La riga pipe vuota richiesta dalla sintassi è distinta dal corpo. Sono state completate le 11 caption di tabella nelle nove trascrizioni interessate; canonici, asset e render non sono stati rigenerati.

L'eccezione `.gitignore` riguarda esclusivamente i PDF sintetici sotto questa directory: tutti i venti PDF canonici, fonti e derivati sono versionabili. Le receipt V della revisione precedente restano storiche; le nuove qualificazioni richiedono M e una receipt derivata di invarianza del contenuto già approvato, senza una nuova conferma umana.
