# Review — implementazione — run-a002-fase1-corpus — r001 — Claude

- Autore/provider/modello e chat: Claude, provider Anthropic, modello Claude Opus 5.5
  (`claude-opus-5-5`, contesto 1M), eseguito tramite l'estensione Claude Code per VS Code.
  ID chat non esposto. Il reviewer previsto è Claude: nessuna sostituzione.
- Prompt di origine: `prompts/08-review-implementation-r001-claude.md`,
  SHA256 `5eb5924e1b14bae8effc7014e730656351f0f6bf8f2b62ed967170292936a814`.
- Oggetto e revisione esatta: implementazione r001 completa (mandato05 + addendum V + prompt06),
  piano r001 `3ac23bce…dd4ab8` ed errata r001 `a09cbb8b…e0dfd9`, entrambi verificati.
- Snapshot, HEAD e impronta verificati prima/dopo: `snapshots/implementation-r001.json`
  SHA256 `6d843889bd8de41b2b6c0842fe6377ba14b9145cc30bbf722662ec77888aa911`, impronta
  `457568ae…7169fb70`. Verify **prima** MATCH/exit0. Verify **dopo** MATCH/exit0.
  Argv, cwd e output sono in `evidence/review-implementation-r001-claude/verify-{before,after}.txt`.
- Indipendenza: non ho letto la review ChatGPT, le sue evidenze o il suo checkpoint, né alcun
  arbitrato implementativo, che non esiste. Ho letto l'arbitrato del piano r002, come autorizzato.
  Nessun agente delegato.
- Giro: iniziale e completo sul delta.
- Base iniziale e delta: branch `feature/run-a002-corpus`. HEAD, `dev`, `origin/dev` locale e
  merge-base coincidono su `80c31da46e5caab990d3596c79c44e524ec1dc95`. Indice vuoto.
  Il delta sta nel worktree: 11 tracciati modificati (tra cui `run_offline.py` e indici docs)
  e file nuovi (28 sotto `scripts/benchmarks/`, 3 test, 2 guide, 3 report benchmark,
  109 file corpus non ignorati, documenti a001 ereditati). Converter, pin e lock non sono modificati.
- Esito: **NO_GO**.

## Ambito e prove

Ho usato le skill `manage-implementation-run` e `verify-conversion-fidelity`.

**Letture.** AGENTS, `temp/HANDOVER`, sezione corrente di PROJECT-CONTEXT, STATE/HANDOVER/brief
della run, prompt08 e protocollo con template. Il checkpoint del supervisore. Piano ed errata
E01–E08 integrali, arbitrato r002, addendum V, prompt05 e prompt06. I report dell'autore
correnti (`implementation-r001.md`, `-v2-fix.md`, checkpoint autore nella parte corrente).
I tre report in `documentation/benchmarks/` (verified-reference, human-comparison, native come
storia). Guide how-to/reference e `PROTOCOL.md`.

**Codice letto integralmente:** `data`, `safety`, `audit`, `profile`, `adapters`, `literal_child`,
`legacy_child`, `resources`, `benchmark`, `campaign`, `launch`, `observations`, `metrics`,
`math_structure`, `document_response`, `verification`. Ho letto inoltre la parte iniziale di
`reference_check` e gli import di tutti gli altri moduli. Il diff di `run_offline.py` e la sua
funzione `monitored_command`/`inside`. `PandocConverter` legacy. I nomi di tutti i test e i test
mirati sul parser V, sugli unknown e sulla sicurezza.

**Prove esaminate:**
- **V:** `V-comparison-001` (indice, C02 HTML/.md, packet.json e identità), risposta originale
  e metadati, `V-response-document-001/response.json`, `verification-document-001`
  (summary e record C02).
- **M:** M-gold-006 (403 record required, tutti PASS) e M-controls-004 (89 PASS), suddivisi per
  `evidence_kind`.
- **Caption:** caption-source-contract-001.
- **Corpus:** provenienza delle generazioni 013/014 nel manifest (88 file su 90 byte-identici,
  compresi i PDF canonici; differiscono solo i due PDF LibreOffice).
- **Stage040 e receipt R:** receipt R e `inside` di tests-fast-004, tests-comparison-006,
  document-response-001, document-verification-001 e rescore-002 (tutte PASS, `--net=none`,
  preflight exit0, snapshot MATCH prima/dopo). Stdout della suite: `304 passed, 310 subtests`;
  mirati: `163 passed`.
- **Campagne e rescore:** contracts-003 (20 casi, tutti conformi, H15 preempted conservato);
  R-comparison-visual-001 (FAIL conservato); campagne 002/003 (raw, run, observations, metrics
  di attempt legacy e diretti); rescore-001 (32 metriche) e il suo input con i sei delta helper;
  ledger, entry e budget-checkpoint-008.

**Comandi miei, tutti read-only stdlib, con output nelle evidenze:**
- `run_context.py verify` prima e dopo.
- Identità Git, `git diff --check` e controllo dei link locali delle guide/report (nessun link rotto).
- `git check-ignore` sui PDF del corpus.
- `static_checks.py` (script mio) → `static-checks.json`. Ricalcola i 201 hash degli input esterni
  di stage040 contro il worktree (**0 mismatch**) e gli hash del manifest (10/10 coincidono).
  Controlla anche didascalie, markup header delle tabelle e nodi unknown.

**Non eseguito:** nessun import o esecuzione del codice prodotto o benchmark, nessuna suite,
build, installazione, conversione o R. Le riproduzioni dinamiche dei rilievi I003/I004/I006 non
sono state fatte. Per queste conclusioni basta l'analisi statica del codice, citata per riga, e
le prove sotto indicano il test di chiusura. Non ho rieseguito la campagna a001 né i workload
già attribuiti.

## Rilievi

| ID | Severità e blocco | Posizione | Raggiungibilità/input ammessi | Attribuzione e requisito | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLA-I001 | Alta, **bloccante** | `.gitignore:62` `*.pdf`; `tests/fixtures/conversion-corpus/inputs/C06…C10.pdf`, `assets/renders/C02…,C04…pdf`, 13 `sources/derivations/*.pdf` | Certa. L'integrazione Git manuale dell'utente, prossimo passo dopo il GO, ometterebbe i file senza alcun avviso | Delta a002 (posizione del corpus) che urta una regola ereditata. Violati A02, brief criterio 1, piano §2 («fixture canoniche consegnate hanno sempre identità byte precisa») e §10 («corpus versionato offline») | 5 canonici su 10 (tutte le 9 pagine PDF), i render PDF LibreOffice usati da M e 13 derivazioni sono ignorati. Restano fuori dall'impronta dello snapshot comune e dal futuro commit. In un clone pulito M, packet, verification e `test_compile_actual_text_partial_history_and_changed_include` (`comparison.build` copia tutti gli originali) fallirebbero. La run successiva Docling/Marker, che ha bisogno proprio dei PDF, non avrebbe il corpus. Nessun report lo segnala | `gitignore-pdf-check.txt` (20 PDF ignorati, hash = manifest); `git-identity.txt` (109 file corpus non ignorati, nessun PDF) | Regola di eccezione (es. `!tests/fixtures/conversion-corpus/**/*.pdf`) o `.gitignore` locale. Prova: `git check-ignore` vuoto per i 20 PDF, `git status -uall` che li elenca, hash invariati rispetto a manifest/stage040. Nuovo snapshot comune del supervisore. Nessun rerun se i byte restano identici. Nota nel README del corpus/guida |
| CLA-I002 | Alta, **bloccante** | `metrics.py:156-164` (celle per coordinata con la riga 0 = header pipe) e `:138-146` (numeri delle celle per `text_index`); gold `cells` senza semantica di header; sorgenti C02/C04/C05 senza markup di header | Certa, su tutti gli output effettivi di 5/8 percorsi; per costruzione su qualunque motore che converta una tabella senza header in pipe table | Nuovo codice e corpus. Piano §2 r146 («La tabella deve distinguere header…»), §4 tabella celle, I02/I06 («Celle… denominatori e localizzazione corretti»), brief (misure per fenomeno non viziate) | Le tabelle DOCX/RTF/TeX non hanno header (nessun `w:tblHeader`, `\trhdr` o separatore); solo l'EPUB usa `<th>`. Il gold non dichiara header. La sintassi pipe richiede una riga header, quindi Pandoc emette un header vuoto e poi il corpo **identico** al gold. Lo scorer allinea la riga header sintattica alla riga 0 del gold. Risultato pubblicato: **0/9, 0/6, 0/9 celle**, 4+4+2 numeri «persi» (DOCX 8/13, TeX 4/9, RTF 3/8), con `actual: []` su numeri presenti. Senza l'artefatto: celle 9/9, 6/6, 9/9 e numeri DOCX 12/13, TeX 8/9, RTF 5/8. La metrica premia inoltre la promozione di una riga del corpo a header (alterazione semantica) e penalizza la resa fedele. Il supplemento human-comparison afferma «perdite di shape/contenuto», il native report che «TeX… cambia celle». La qualità FAIL resta per altre perdite reali, ma la misura per fenomeno e la sua attribuzione al motore sono viziate | `static-checks.json` → `table_header_artifact` (`observed_first_row_all_empty: true`, `observed_body_equals_gold_grid: true`), `source_table_header_markup`, `gold_table_header_fields: []`; raw `campaign-003/direct-docx/…/document.md` | Semantica di header esplicita e verificabile: `header_rows` nel gold, derivato dal sorgente e controllato da M sul markup (`w:tblHeader`/`\trhdr`/`th`). Scorer con regola documentata che confronta presenza/contenuto dell'header separatamente e il corpo per coordinate, senza riallineamenti arbitrari. In alternativa, sorgenti con header marcato e nuove generazioni. Test con tre casi: tabella headerless resa con header vuoto → corpo PASS; promozione indebita → diff header; perdita reale di cella → FAIL. Rescore dei 32 raw senza motori e correzione dei report. Se cambiano gold o vista, M/V dipendenti secondo l'arbitrato |
| CLA-I003 | Media, **bloccante** | `metrics.py:102` (`if missing and not unknown`), `:197` (`missing_page … and not unknown`); contrasto con `PROTOCOL.md:87` e piano r321-322 | Su input ammessi: basta **un** nodo unknown qualsiasi (tabella HTML con span, `\comando` raw, MathML, quote complessa, classi già previste dal reader). Molto probabile con i motori della run successiva. Non presente nei 32 output attuali (verificato) | Nuovo codice. Piano §4 («Un FAIL critico ha precedenza sul non misurato, con entrambi esposti»; perdita di pagina distinta), A06, I05/I06 | Con un unknown anche estraneo, i blocchi mancanti non entrano in `failures` e `missing_page` resta false. Una pagina intera persa diventa INDETERMINATE/exit5 invece di FAIL/exit3: la perdita critica è declassata e la perdita di pagina non è rilevata. Nessun test copre il caso unknown + perdita | Codice citato; `static-checks.json` → `rescore_unknown_vs_missing` (nessun unknown attuale) e `source_lines` | Il mancante resta FAIL salvo prova esplicita che un unknown possa contenerlo (regola versionata, es. similarità del testo del blocco col raw unknown). Solo quel blocco diventa unresolved. `missing_page` non dipende da unknown estranei. Test sotto R/suite: controllo K02 senza pagina 2 più un paragrafo `\unknown{x}` → exit3 con `missing_page` p2; unknown che contiene il blocco → unresolved, non FAIL |
| CLA-I004 | Media, non bloccante | `observations.py:46-58,107` (il titolo resta nel target) e `:258` (regex figure); `:79-90` (`` `…`{=latex} `` letto come codice) | Sintassi Markdown di base (titolo link/immagine) e raw inline previsti dal profilo E05 (`raw_attribute`); il raw inline compare nell'output TeX effettivo | Nuovo codice; E05 («Sintassi base … link/image con offsets»; raw_attribute → tipo conservato o UNAVAILABLE) | `![a](<../../../../../temp/run-a002-fase1-corpus/reviews/m.png "t">)` → target `m.png "t"` → falso MISSING_RESOURCE (run exit2, bundle INCOMPLETE) o UNSAFE se il titolo contiene `:`. Per i link → falso FAIL di riferimento. `` `\cite{bib1}`{=latex} `` diventa testo di codice e conta come errore di testo invece di raw TeX unknown. Nessun falso PASS, ma esiti operativi e localizzazione errati | Codice citato; output `campaign-003/direct-tex/…/document.md` e failure b13 nella metrica TeX | Separare titolo e target secondo CommonMark; trattare il raw inline per formato come i fence raw. Test positivi e negativi P-autonomi; rescore se cambia un'osservazione esistente |
| CLA-I005 | Bassa, non bloccante | `generate_corpus.py:629-634`; `gold/*/transcript.txt` | Su tutti i gold correnti | Nuovo corpus; piano §3 («transcript.txt contiene il testo completo in ordine logico») | Le **11 didascalie di tabella** in 9 documenti mancano da `transcript.txt`: è la stessa classe di difetto della caption C02. Metriche, M e vista V usano `annotations.json` e non ne sono toccate; un consumatore futuro del transcript avrebbe il testo incompleto | `static-checks.json` → `transcript_missing_captions` | Includere `caption` nel generatore, oppure dichiarare il transcript vista parziale. Nuova revisione dei soli transcript con nota e inventario aggiornato |
| CLA-I006 | Media-bassa, non bloccante | `adapters.py:88,104` (solo hash di `/usr/bin/pandoc`); `legacy_child.py` e `convert()` senza controllo del PATH; nessun test con PATH falso | Con un environment di stage diverso da `PATH=/usr/bin`; nelle campagne effettive il PATH era `/usr/bin` in un environment hashato e protetto | Nuovo codice; E08 («risolvere il token pandoc secondo quel PATH… prima del figlio e dopo ogni tentativo»; chiusura «PATH con Pandoc falso… non cambia il resolved hash») | L'identità del binario usato dal legacy vale **per costruzione** dell'environment, non perché il contratto la verifichi. Non vengono registrati PATH validato, token risolto e realpath, e manca la prova di chiusura E08. Gli attempt eseguiti restano attribuiti correttamente | `campaign-003/environment.json`; `run.json` legacy (`pandoc_before/after` su path fisso) | Prima e dopo il figlio: allowlist del PATH e `shutil.which('pandoc', path=…)` → realpath/hash uguali a config, registrati. Test con `pandoc` falso anteposto al PATH → exit4/H10 |
| CLA-I007 | Bassa, non bloccante | `rescore-verified-reference-input-001.json` (`historical_converter_helper_delta`) | Riguarda i 12 raw riusati TXT/EPUB e le prove storiche H06/H08 su stage034 | Provenienza (AGENTS: «conservare sorgente…»; I09) | Dei sei delta storici (`safety.py` 9690→9741 B, `adapters.py` 7615→7623 B) restano solo gli hash, non i byte, e nessuna nota ne descrive il contenuto. Il riuso è comunque motivato: l'output dipende da Pandoc, argv, input e binario, tutti invariati. La sua ammissibilità resta però dichiarata e non ispezionabile | File citato; storia hash per stage in `stage-s0*/benchmark-inputs.json` | Descrivere il delta (per esempio dal diff della chat autrice, se disponibile) oppure rieseguire i 12 attempt TXT/EPUB e i casi H06/H08 sul codice corrente (costo minimo). In futuro conservare i byte degli helper consumati |

Lacune già autorizzate e non contestate:
- L01: conversioni PDF/OCR NOT_MEASURED.
- L02: compilazione TeX NOT_MEASURED.
- L04: giudizi tipografici e semantici degli output UNAVAILABLE, non assegnati.
- L05: CPU lower bound, RSS campionato, disco retained, cache OS non controllata.
- Renderer Chrome/EPUB R NONPASS conservato e non sanato.
- I link dei report permanenti verso `temp/` richiedono l'archiviazione prima della pulizia,
  come già dichiarato dall'autore.

## Motivazione dell'esito e limiti

Criteri controllati:

- **I01.** Il perimetro è rispettato: nessuno schema applicativo, nessun motore scelto, nessuna
  fase2. L01–L05 sono espliciti. I requisiti di corpus e tabelle sono però realizzati solo in
  parte (I001, I002).
- **I02.** Copertura 10/9/12 con IT/EN, misto C10 e controlli; licenze e notice presenti.
  Due generazioni con distinzione corretta tra byte-identici e PDF LibreOffice. La consegna
  permanente non è completa nel set tracciabile (I001). La semantica di header è assente (I002).
- **I03.** M è indipendente (import di sola `data`/`safety`/`resources`/`png_codec`) e locale sul
  canonico; caption discriminante con vecchia revisione FAIL e nuova PASS. Su 403 record, 220
  sono controlli di copertura delle annotazioni e 134 encoded: proporzione dichiarata e coerente.
  M non controlla il markup di header (I002).
- **I04.** Il packet V non contiene output dei motori e usa parole semplici. Il riferimento è
  completo, incluse caption, MathML, note e celle vuote. La risposta originale è conservata
  byte-esatta. La proiezione collettiva è dichiarata e non inventa righe per documento.
  `validate_response` riclassifica i byte; i negativi su vaghezza, eccezioni, sottoinsiemi e
  stale sono testati. Il parser è stato esteso a posteriori con la frase effettivamente ricevuta:
  è accettabile perché la forma è finita, non amplia l'ambito e la seconda frase non è
  interpretata come approvazione degli output. Unico limite: la vista mostra come header una riga
  che il sorgente non marca (I002).
- **I05.** Observe è gold-blind, con raw/offset e digest. Alcune lacune sulla sintassi di base
  (I004). Matching one-to-one e ambiguità sono corretti; il mascheramento da unknown è un difetto (I003).
- **I06.** Numeri lessicali, meno solo in matematica e AST E01 coerenti con FP/FN. Le celle sono
  viziate (I002) e la precedenza del FAIL non è rispettata (I003).
- **I07.** Positivi e negativi sono ampi, ma mancano tre casi: tabella senza header, unknown con
  perdita, PATH falso.
- **I08.** Stati, exit, H01–H16 e input unsafe (exit4) distinti da output unsafe (exit2) sono
  corretti nel codice e nelle receipt storiche.
- **I09.** Gli 8 percorsi × 4 sono reali; A10 è soddisfatto (figura DOCX con SHA esatto,
  referenziata). Riuso 12 + 20 motivato (I007). I numeri per fenomeno sono viziati (I002).
- **I10.** Validazione prima dei tool, no-follow, assenza di fetch, `shell=False`. R non è
  presentato come sandbox del filesystem.
- **I11.** Stage040, i dieci hash, preflight e R stesso interprete sono PASS. Gli input
  consumati coincidono byte per byte con il worktree attuale (verificato indipendentemente).
  Il fix post-exit del runner è corretto e conserva i vecchi falsi verdi. E08 è parziale (I006).
- **I12.** Ledger coerente (41737.61 s, 1465.67 CPU-s lower bound, 465182720 B, €0, 0 byte),
  nessun reset. Guide e link sono validi; il set Git non è consegnabile (I001).

I tre rilievi bloccanti riguardano la consegna permanente del corpus e la correttezza delle
misure per fenomeno: non sono preferenze. Le correzioni restano locali (`.gitignore`, gold o
scorer delle tabelle, precedenza dei mancanti) e non richiedono conversioni nuove: bastano
rescore, test e report. Non ricertifico i risultati storici e non attesto prove che non ho
eseguito. Non chiedo motori perfetti: i FAIL reali di Pandoc (RTF `蜢?7`, caption, anchor,
note) restano risultati validi.
