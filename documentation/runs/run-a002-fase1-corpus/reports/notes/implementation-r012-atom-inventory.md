# Inventario tecnico r012, prima delle modifiche

Ruolo implementatore; mandato40 F50–F53, nessun nuovo piano o GO.
v11 fisico preservato in `work/implementation-r012/r011-observations.py`.

| Punto | Dominio e decisione v11 | Dati ricevuti dal consumatore / intervento |
| --- | --- | --- |
| `source_scope_end`, `inline_scope_end`, `raw_fence_end` | Sorgente fisico; limiti di producer, raw fence e barriere di blocco | Limiti numerici; preservare i confini, senza ridurli dopo stripping |
| `html_lexeme`, `html_tag_boundary` | Token opachi e tag/attributi, tokenizer pinned/F36 | Intervalli fisici e eventi; nessun nuovo feed puro |
| `code_end`, `metadata_at`, `math_end`, `command_at` | Primitive di atomo nel limite esplicito | Oggi richiamate indipendentemente da scanner/label/inline/delimiter/pipe; introdurre descrittore comune tipo/start/end/body/metadata |
| `markdown_label`, `markdown_link` | Label nel dominio del producer; LF via code/token/escape; destinazione distinta | Oggi ledger code senza metadata e nessun ledger LF escape; registrare gli atomi completi e il primo bound LF non posseduto una volta |
| `markdown_lexemes` | Sorgente fisico o foglia proiettata; owner HTML prevale sugli atomi Markdown | Oggi riconoscimento ripetuto e salto code label prima di metadata; produrre decisioni riusabili e coverage esplicita |
| `markdown_units`, `strip_inline_end`, `peel_attrs`, `pipe_cuts/row` | Righe fisiche, trim fuori token, heading/celle | Conservare barriere source-proven; consumer della decisione di atomo, non nuova semantica |
| `observe/read_inline/read_parts` | Paragrafo/heading/caption/cella oppure quote/list/note senza prefissi | Stringa, barriere e mappa char; riconoscere fence nel dominio di blocco del container prima del walk inline |
| `inline_html_events/validate_html` | Aperture reali e raw HTML esatto nel dominio deciso | Guard separata da albero, eventi ordinati; visita tutte le aperture prima dell'errore |
| `inline`, ricorsioni label/span/formatting, `inline_delimiter` | Foglie e sottodomini, code/math/HTML | Oggi riscoprono code/math su stringhe e limiti diversi; riusare decisioni ritagliate/traslate, comprese barriere e metadata |
| `HTMLGuard/HTMLTree/html_nodes` | HTML puro già posseduto dal raw o tag | Tokenizer unico; guard indipendente dall'errore/dimensione dell'albero, residue unsupported |
| `observe/node` fallback | Raw del blocco e foglie mappate al sorgente | Oggi presenza segments esenta indiscriminatamente; sostituire con copertura del dominio esplicita, residuo non proiettato diagnosticato |
| `audit/observe_bundle/score/CLI` | Solo osservazioni emesse | Nessuna patch compensatoria: reference/unknown devono arrivare dal reader; preservare fallback IDv1 e qualità INDETERMINATE |

Scelta circoscritta: descrittori interni nel reader esistente, senza nuovo
modulo/package/schema pubblico. Il walk decide gli intervalli; guard e
proiezione ricevono quella decisione. La label decide una volta le LF ammesse
e le barriere opache; un indice evita la scansione ripetuta di tutte le LF/code.
Fence esatto nel container: guard HTML attivo e diagnosi unsupported esplicita
se non proiettato; fence ordinario resta letterale. Source-map costruita dai
producer, senza ricerca del testo per ricostruire prefissi.

Prove previste: generatore finito separato, attesi manuali di contenuto e
coordinate prima del workload, confronto fisico v11, regressioni r009–r011,
S/B/I/R nuovi e prove finali attribuite. Token/comment/CDATA, F31, F36/F40/F41,
owner HTML e metadata sono barriere/contratti da preservare. La presenza di
segments non certifica la guardia. CPU lower/RSS sampled, nessun upper;
deadline rimossa, altre quote invariate. Stato/arbitrati restano al supervisore.
