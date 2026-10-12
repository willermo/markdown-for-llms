# Riferimento verificato e consegna a002 — 2026-10-08

**Tutti i 13 documenti sono VERIFIED; A03/L03 è chiuso.** L'utente della chat
implementatrice ha confermato la corrispondenza completa del confronto
`V-comparison-001`. M required PASS, completezza della vista e risposta V reale
coincidono per sorgenti, gold, coppia e ambito. L'implementazione è consegnabile
al supervisore per ricezione, snapshot comune e due nuove review indipendenti
ChatGPT/Claude. Questa consegna dell'autore non è un GO sul codice.

Questo esito aggiorna il
[supplemento precedente, con V allora pendente](2026-10-08-a002-human-comparison.md)
e il [report storico della campagna iniziale](2026-10-08-a002-native.md), entrambi
conservati immutati. Dettagli del corpus revisione 4, fix didascalie, due
generazioni reali e metriche dei 20 tentativi dipendenti restano nel supplemento.

## Risposta effettiva e ambito

Testo ricevuto, conservato esattamente, compresi spazi e articolo attaccato:

> Tutti  idocumenti corrispondono. Ad un esame visivo, valido le conversioni verificate.

[Originale](../../temp/run-a002-fase1-corpus/evidence/implementation-r001/V-received-chat-001.txt),
[metadati di registrazione](../../temp/run-a002-fase1-corpus/evidence/implementation-r001/V-received-chat-metadata-001.json),
[receipt di importazione](../../temp/run-a002-fase1-corpus/evidence/implementation-r001/V-response-document-001/response.json).
SHA256 originale `2b34e76549d084988bb09ba56d121e66b0b20d81443e2922f01ef1a3cda87f9e`.
Packet SHA256 `ac0921defac79ca063ef89fe270ed09e8d991e7dc17d9a0aeae3c2c4c06a5c6f`.

Autore attribuito al ruolo effettivo «utente della chat corrente»; nessuna
autenticazione crittografica o nome personale ricavato dal filesystem. Ora
UTC della prima registrazione locale **2026-10-08T18:29:05.249323+00:00**;
timestamp della piattaforma indisponibile. L'utente dichiara un esame visivo,
ma non ne fornisce inizio o durata: entrambi restano non misurati. I 260 minuti
del vecchio workflow e la stima prospettica 60–90 minuti del nuovo confronto
non diventano tempo attivo o una nuova durata concordata.

La dichiarazione «Tutti … documenti corrispondono» si riferisce alle tredici
coppie complete presentate nella chat: dieci documenti corpus e tre controlli.
L'importatore conserva **una sola dichiarazione originale collettiva**, con
gli ID del packet. Ogni decisione documentale dichiara la proiezione di tale
ambito; nessuna riga `Cxx: corrisponde`, firma per blocco o 215 giudizi atomici
è stata inventata. La seconda frase conferma l'esame del materiale mostrato;
non qualifica gli output dei motori, separati dalla vista di riferimento.

L'importatore riconosce un insieme finito di dichiarazioni complete, tollerando
gli spazi e «idocumenti» soltanto per l'interpretazione. Byte originali immutati.
Approvazioni vaghe, eccezioni, sottoinsiemi e dubbi non accettano collettivamente
il packet. La verifica riclassifica i byte ricevuti e controlla anche l'ambito
della dichiarazione; le risposte per documento e il formato storico v1 restano
disponibili.

## Verifica effettiva

[Summary documentale](../../temp/run-a002-fase1-corpus/evidence/implementation-r001/verification-document-001/summary.json)
schema2: `all_verified=true`, tredici documenti VERIFIED, M PASS e V ACCEPTED.
La compilazione ricontrolla sorgenti/include/render/asset e vista completa
contro il gold corrente, senza riscrivere annotazioni o il packet.

M-gold006 aveva **403 required PASS**, M-controls004 **89 required PASS**;
il checker indipendente aveva 13 documenti e 1906 controlli PASS. La risposta
reale chiude la verifica visibile mancante, senza attribuire all'utente i
singoli controlli meccanici. Gli stati PROVISIONAL nelle annotazioni manuali
restano storici immutabili; lo stato qualificato è nella receipt separata,
legata ai medesimi hash.

Nuovo **stage040** dopo il cambiamento del parser: ricostruzione sdist/wheel,
reinstallazione della wheel canonica, S/B/I, dieci hash, inventario esterno e
preflight effettivi. R invariato, Firejail e rete disabilitata; nessun criterio
del runner modificato. Tutti i nuovi workload hanno R PASS:

| Workload effettivo | Risultato |
| --- | --- |
| tests-comparison-006 | 163 test PASS, 6.77 s |
| tests-fast-004 | 304 test e 310 subtest PASS, 20.21 s |
| document-response-001 | Una dichiarazione reale, 13 ACCEPTED |
| document-verification-001 | 13 VERIFIED, sorgenti e viste coerenti |
| rescore-verified-reference-002 | Scoring di 32 output conservati, nessun motore avviato |

La suite veloce usa le esclusioni prescritte API/packaging/Docker; i due smoke
radice sono esclusi. Non è il collaudo completo o una certificazione della
migrazione. I nuovi test distinguono dichiarazione collettiva completa da
approvazione vaga/parziale, eccezioni, ambiguità e identità STALE; verificano
testo originale, ambito alterato e compilazione positiva di tutti i documenti.

## Metriche aggiornate senza nuove conversioni

[Input congelati dello scoring](../../temp/run-a002-fase1-corpus/evidence/implementation-r001/rescore-verified-reference-input-001.json)
prima del workload; [risultato effettivo](../../temp/run-a002-fase1-corpus/evidence/implementation-r001/rescore-verified-reference-001/summary.json).
Sono 12 output su input/gold invariati della campagna002 (TXT e due EPUB)
più 20 output dipendenti della campagna003 (DOCX/RTF/TeX diretto e DOCX/RTF
legacy). Gli otto percorsi conservano quattro tentativi ciascuno. Non è una
nuova campagna di conversione; i final originari restano immutati.

Ogni score è un processo separato, con argv, exit e risorse reali. **356 file
protetti** sono identici prima/dopo: raw, observations, audit, metriche/final
originari, configurazioni e gold. Input effettivi e include sono confrontati
con i sorgenti correnti. Denominatori, totali, alignment, errori, ordine,
CER/WER e asset verificati sono identici alle metriche originali. Cambia solo
la disponibilità della qualificazione del gold, ora VERIFIED.

| Percorso | Score dei quattro output | Qualità dopo V reale |
| --- | --- | --- |
| direct-txt | 0, 0, 0, 0 | 4 PASS, senza formule o proprietà output non assegnate |
| direct-docx | 3, 3, 3, 3 | 4 FAIL |
| direct-epub | 3, 3, 3, 3 | 4 FAIL |
| direct-rtf | 3, 3, 3, 3 | 4 FAIL |
| direct-tex | 3, 3, 3, 3 | 4 FAIL |
| legacy-docx | 3, 3, 3, 3 | 4 FAIL |
| legacy-epub | 3, 3, 3, 3 | 4 FAIL |
| legacy-rtf | 3, 3, 3, 3 | 4 FAIL |

Totale **4 PASS, 28 FAIL**. Le conversioni erano SUCCESS/COMPLETE; il batch
di qualità resta FAIL. Le perdite note non sono sanate dalla conferma del
riferimento. L04, tipografia e semantica matematica degli output, resta
UNAVAILABLE/non assegnata per i percorsi con formule.

Le configurazioni dei vecchi TXT/EPUB includono hash storici di helper poi
cambiati per il fix locale (`safety.py`, `adapters.py`): sei delta esplicitamente
registrati negli input dello scoring. Non sono prove di conversione con il
codice corrente. Il nuovo scorer legge gli output realmente preservati;
l'identità del riferimento e tutte le misure strutturali sono controllate.

Un primo controllo amministrativo pretendeva erroneamente che ogni helper
storico coincidesse con il codice corrente; fallì prima di congelare o avviare
lo scoring. È conservato in `rescore-freeze-admin-error-001.json`. Il primo
snapshot dello scoring fallì su un artifact corpus fuori dalla run; target e
receipt `rescore-verified-reference-001-snapshot*` sono conservati. L'orchestratore
proprio è corretto per registrare come artifact solo file della run, mantenendo
tutti gli input esterni nel target. Il nuovo snapshot002 e R002 sono PASS;
nessun fallimento è rinominato o attribuito al motore.

## Costi, limiti e prossimo ruolo

[Ledger cumulativo](../../temp/run-a002-fase1-corpus/evidence/implementation-r001/ledger.jsonl)
dal 07:06:54 UTC, senza reset. Il checkpoint007 misura circa **11 h 29 min wall**,
inclusa l'attesa della risposta; CPU wait4 nota **1465.67 s** come lower bound,
disco retained **465121280 byte** (circa 444 MiB, inodi unici). Nessuna nuova
acquisizione o trasmissione documentale, servizi nuovi **€0**. CPU amministrativa
non monitorata e inferenza non disponibili; disco gross/peak non continuamente
misurato. Quote e riserva originarie restano invariate. Il checkpoint finale
e i documenti di consegna aggiungono pochi byte successivi al campione.

L01 motori PDF/OCR e L02 compilazione TeX restano NOT_MEASURED; L05 sampled RSS
e cache OS hanno i limiti dichiarati. I FAIL R dei renderer Chrome/EPUB già
registrati restano tali: composizione/checker e V reale non li promuovono a
R PASS. Presentazione agent_visual, verifica umana e misure dei motori hanno
receipt e ambiti separati.

Report/checkpoint propri aggiornati; nessuna scrittura Git, registri condivisi,
arbitrato o review simulata. Prossimo ruolo: **supervisore** per verificare la
consegna completa, aggiornare i propri stati e produrre lo snapshot comune per
le due nuove chat indipendenti ChatGPT/Claude. Commit, merge e push restano
azioni dell'utente dopo il GO effettivo.
