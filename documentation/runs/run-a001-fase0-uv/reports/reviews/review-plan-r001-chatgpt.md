# Review — piano — run-a001-fase0-uv — r001 — ChatGPT

- Autore: **Codex, ruolo esclusivo di revisore ChatGPT** assegnato dal prompt utente.
- Provider effettivo: **OpenAI**. Modello: famiglia **GPT-6** indicata dalle
  istruzioni della sessione; identificatore specifico non esposto. Riferimento/ID
  chat non esposto. Non sono inventati identificatori o una diversa interfaccia.
- Data: **2026-10-02, Europe/Rome**.
- Prompt di origine: [review-plan-r001-chatgpt.md](../prompts/review-plan-r001-chatgpt.md),
  ricevuto anche nella richiesta utente.
- Oggetto: [piano r001](../plans/plan-r001.md), letto integralmente; fase 0.1.
- Snapshot comune: [plan-r001.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
- Branch: **feature/run-a001-uv**.
- HEAD, `dev` locale e merge-base pre/post:
  **66ba82200e5def5a4db76f9bafccb0731b506091**.
- Impronta worktree letta dal manifest:
  **111f2830bfbef9b3bf816b02ab8818e8daf2d8a272f37b05e9fc565a004a7b60**.
- SHA-256 piano letto dal manifest e riconfermato sul file:
  **7f9dd0a2ded10c40ef9e3a42ae1424fb08a23059384468ea160dd7173fd3e8fb**.
- Identità: verifica iniziale **MATCH**, verifica post lettura **MATCH**; controlli
  di chiusura in [checks-final.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
  Il worktree presenta le stesse sei modifiche documentali di supervisione.
- Indipendenza: questa chat non ha scritto il piano, svolto supervisione o l'altra
  review. Non letti report, checkpoint o evidenze del revisore concorrente, né
  l'arbitrato corrente; nessun coordinamento o agente delegato. Sono stati letti
  soltanto gli input condivisi autorizzati e fonti primarie pertinenti.
- **Esito: GO sul piano r001 identificato sopra.** Nessun rilievo bloccante;
  un suggerimento opzionale, GPT-P001. Questo GO non autorizza da solo l'implementazione.

## Ambito e prove

La review valuta fattibilità, perimetro e sufficienza delle prove future rispetto
ai sette criteri del brief. Non attesta una migrazione o una conversione eseguita.

### Input letti

- Prompt, `AGENTS.md`, skill `manage-implementation-run`, protocollo e template review;
  handover globale letto per recuperare il contesto, senza adottarne il ruolo supervisore.
- `STATE.md`, brief e prompt di pianificazione; indice architetturale, indice decisioni,
  ADR 0001/0006/0007, roadmap con fase 0.1 e changelog.
- Piano completo, checkpoint del pianificatore, findings, checks e inventario locale;
  manifest comune, catalogo/lista Python e sezioni pertinenti degli help uv congelati.
  I tredici artefatti del registro di pianificazione sono stati riconfermati per
  dimensione e SHA-256. L'exit 1 dei diff contro `/dev/null` identifica file nuovi,
  senza diagnostiche whitespace: non è un fallimento di test della migrazione.
- Preparazione del supervisore e receipt amministrativa, senza usarle come giudizio tecnico.
- `setup.py`, `requirements.txt`, `.gitignore`, `Dockerfile`, `docker-compose.yml`;
  README nelle sezioni installazione, avvio, sviluppo/test e Docker/GPU.
- I dieci moduli flat: `config`, `logging_config`, `exceptions`, `unified_converter`,
  `master_workflow`, `clean_markdown`, `validate_markdown`, `chunk_markdown`,
  `batch_monitor`, `marker_api_server`. Lettura mirata a import, entry point,
  configurazione, avvio delle fasi, output e side effect; inventario AST sull'intero
  contenuto senza importare il codice. Non è un audit di tutti gli algoritmi legacy.
- `tests/conftest.py`, i tre file unitari, integrazione, governance e i due smoke
  manuali alla radice. Consultata la skill `verify-conversion-fidelity` per valutare
  baseline, fixture e confronto dei contenuti.

### Controlli realmente eseguiti

Prima di ogni lettura tecnica sono stati eseguiti i cinque comandi di identità
richiesti: status/branch, HEAD/dev, merge-base e verify `plan-r001`. Esito MATCH,
exit 0 del gruppo. Ripetuti dopo la lettura, con esiti individuali exit 0 e MATCH.
I dati sono in [identity-pre.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
e [identity-post.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).

Sono stati eseguiti letture, hash, analisi AST, confronto delle opzioni con gli
help congelati e soli comandi di versione: **uv 0.10.10, Python 3.12.3, Pandoc 3.1.3**.
La shell non dimostra il funzionamento del candidato Python managed 3.12.13.
Evidenze in [static-checks.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
e [toolchain-static.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
Collegamenti locali degli output e whitespace controllati alla consegna.

Le affermazioni su versioni, sorgenti e contratti sono state confrontate con fonti
primarie tramite browser, registrando data, URL, accessi falliti e inferenze in
[sources.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md). Il metadata PyPI
conferma la presenza di Setuptools 84.0.0 e Requires-Python >=3.10; gli indici
PyTorch espongono wheel cp312 Linux x86_64 delle varianti proposte. Questa conferma
non risolve il grafo transitive. Fonti: [Setuptools](https://pypi.org/pypi/setuptools/84.0.0/json),
[torch CPU](https://download.pytorch.org/whl/cpu/torch/),
[torch cu126](https://download.pytorch.org/whl/cu126/torch/).

**Non eseguiti**, come richiesto: uv lock/sync/build, creazione di ambienti,
installazioni, pytest/unittest o discovery, conversioni, build/up Docker, startup
Marker, download di modelli/font o benchmark. Nessun documento inviato a provider,
nessuna operazione Git di integrazione. I risultati storici 67/6 restano storici.

### Valutazione A1–A7

| Criterio | Passi/prove e valutazione del piano | Prerequisiti, costo, limiti e recupero |
| --- | --- | --- |
| **A1** | P1, V1: pin uv, patch Python e range minor espliciti; managed obbligatorio prevale sulla selezione pyenv della shell. Due ambienti nuovi, inventari e lock invariato sono osservabili adeguati. pyproject dichiarativo e lock risolto hanno responsabilità distinte. | Python/pacchetti acquisiti separatamente, poi download Python vietati. Catalogo non equivale a binario collaudato. Restrizione da >=3.9 a 3.12 dichiarata. Candidato/digest indisponibile torna al supervisore; nessun latest o modifica globale di pyenv. |
| **A2** | P1/P2/P5, V1–V3: quattro runtime, dev separato, server leggero e ML opzionale; dieci py-modules e cinque target espliciti risolvono i difetti reali di setup/config.main. Rimozione delle due vecchie fonti dopo aggiornamento dei consumatori coerente. | Venv e wheel senza ML/dev, metadata e pip check; rete di installazione distinta dai test. CPU/cu126 in conflitto, indici espliciti e marker Linux x86_64 circoscrivono il server senza limitare la wheel base. Git al SHA completo e backend delle build da congelare. Grafo non risolvibile richiede piano server separato approvato, non un'esclusione tacita. |
| **A3** | P2/P3, V3–V5: wheel non editable, CWD esterno senza .py/PYTHONPATH, origini site-packages e binari assoluti; fasi tramite sys.executable -m e controllo moduli installati. config.main mantiene la CLI. Dati, JSON, dotenv e report hanno un CWD definito. | Pandoc e tokenizer pronti; secondi/minuti. Output vuoti e force impediscono che il riuso simuli una fase. Clean/validate eseguiti realmente, help solo dove esiste. Contenuto e negativo exit 1 distinguono avvio corretto da semplice codice 0. Fallimenti con log/output conservati. |
| **A4** | P0/P4, V0/V5/V6/V8: riferimento vecchio distinto dall'output nuovo, F1–F5 e confronti byte/invarianti su testo, formule, numeri, riferimenti, ordine e asset. F2 conserva le perdite legacy; esclusioni limitate a campi deterministici identificati. Suite selezionata, nuovi test, governance e mock sono distinti. | Cache tokenizer preparata con hash; guardia rete in processo e fixture dei subprocess senza HTTP. Smoke radice esclusi anche da pytest .; non basta testpaths. Baseline impedita e danno nuovo restano aperti prima del GO finale. Nessuna normalizzazione globale o skip per replicare numeri storici. GPT-P001 rafforza facoltativamente il caso multi-chunk. |
| **A5** | P7/V9: modifiche mirate alle istruzioni uv, installazione, test, avvio, Docker/GPU e tre guide italiane Diátaxis; eliminazione di Poetry/requirements/setup obsoleti e comandi inesistenti. Nuova applicazione correttamente futura, wrapper legacy identificato. | Comandi nel profilo dichiarato, link e diff verificabili; niente avvio real-engine come effetto di V9. Istruzioni d'inferenza non provate etichettate. Copertura editoriale adeguata alle sezioni interessate, senza riscrittura generale. Changelog del realizzato distinto dai passaggi del supervisore. |
| **A6** | P5/P6, **V7/V8 obbligatorie**: immagine CPU costruita dal lock, package non editable, digest reali, native congelate e contesto filtrato dimostrato con sentinelle; probe offline su import/provider/firma/output reale con converter fake e constructor patchato. Il candidato Marker ha motivazione statica pertinente. | Build con ML può richiedere GB/rete/minuti e mandato operativo esplicito; non è un test veloce. Native/digest/cache non ancora verificati. V7/V8 impedite mantengono A6 aperto. V10/V11 rinviabili solo senza dichiarare Marker/GPU operativi, registrando assenza e limiti; incompatibilità o cambi semantici del wrapper richiedono rivalutazione dal supervisore. |
| **A7** | P0/consegne/V12: quattro report reali, due arbitrati e snapshot pre/post nei due giri, ruoli separati e trattamento di tutti i rilievi. Un'esecuzione tecnica non sostituisce review/arbitrato. | Revisori previsti ChatGPT/Claude, sostituzioni registrate. Cambi sostanziali ritornano al supervisore. Commit/merge/push e promozione manuali dell'utente dopo GO finale. Questa review consegna soltanto il proprio giudizio sul piano. |

### Sufficienza del catalogo V0–V12

V0 conserva fixture/hash, versioni baseline, output e fallimenti attesi; la sua
assenza non può essere rimpiazzata dal risultato nuovo. V1 ricrea due ambienti
base/dev; V2 ispeziona wheel e sdist e verifica backend; V3 installa la wheel con
runtime dal lock senza nuova risoluzione. I prerequisiti di download/build sono
espliciti e restano separati dall'esecuzione dei test.

V4 esercita le CLI corrette; V5 avvia davvero cleaning, validation e chunking e
conversione Pandoc su input sintetico. Hash, byte/invarianti, report/index e codici
0/1 sono osservabili adeguati. I side effect di import su log richiedono CWD
scrivibile e controllo delle destinazioni. La prova editable o i sys.path della
suite legacy non sostituiscono V3/V5.

V6 richiede suite selezionata, due discovery, governance distinta, preflight cache
e controllo dei test senza rete/modelli. V9 lega i comandi delle guide ai profili
effettivamente provati. Entrambe conservano conteggi/esiti reali senza derivarli
dalla storia del bootstrap.

V7 è una build effettiva e controlla anche ciò che entra nel contesto: la sola
lista COPY non basterebbe. V8 dimostra import e contratti offline con sostituzioni
esplicite, non l'inferenza. Il constructor upstream scarica un font: il piano lo
riconosce e richiede patch prima della costruzione. [BaseConverter](https://raw.githubusercontent.com/datalab-to/marker/v1.10.2/marker/converters/__init__.py).

Il collegamento release/commit, la firma del converter e il modello MarkdownOutput
sostengono la scelta candidata. Il wrapper locale ha già il ramo .markdown,
ignorando però immagini e varie opzioni; il piano conserva e documenta questi
limiti. Non occorre provare l'inferenza per giudicare implementabile una migrazione
della toolchain con questo perimetro. Fonti: [release](https://github.com/datalab-to/marker/releases/tag/v1.10.2),
[PdfConverter](https://raw.githubusercontent.com/datalab-to/marker/v1.10.2/marker/converters/pdf.py),
[MarkdownOutput](https://raw.githubusercontent.com/datalab-to/marker/v1.10.2/marker/renderers/markdown.py).

V10 è necessaria prima di dichiarare conversione Marker reale; richiede startup,
pesi/font identificati, PDF valido e formato non PDF, Markdown confrontato con il
sorgente, errori e asset/limiti. V11 aggiunge build GPU e dispositivo effettivo:
TORCH_DEVICE o disponibilità della wheel non bastano. V12 resta amministrativa
e indipendente dalle prove di conversione.

La registrazione comune di comando/CWD/profilo/hash/exit/output e stati PASS,
FAIL, NON_ESEGUITA, IMPEDITA è sufficiente a consegnare anche risultati negativi.
Recupero circoscritto agli ambienti/container di prova, conservazione degli input
e ritorno al supervisore impediscono che un prerequisito mancante diventi un GO
fittizio. I costi sono stime da misurare, non risorse già disponibili.

## Rilievi

| ID | Severità e blocco sì/no | Posizione | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- |
| **GPT-P001** | **Bassa; blocco no. Suggerimento opzionale di copertura.** | Piano r001, F1–F5 (righe 323–329), confronto chunking (340–344) e V5 (591–607). | Esplicitare una fixture abbastanza lunga da produrre più chunk. Le caratteristiche minime attuali garantiscono contenuto per validation, ma non richiedono due chunk; l'osservabile overlap potrebbe quindi non essere esercitato dalla prova CLI esterna. | V5 usa chunk-size 1000; F1 richiede contenuto sufficiente per validation senza minimo di token. Il CLI in chunk_markdown.py può anche cambiare strategia per chunk troppo grandi. La suite contiene già casi lunghi, ma usa import dai sorgenti e non sostituisce la prova installata. | Se accolto, aggiungere un caso sintetico deterministico con almeno due chunk verificati nella baseline e nella wheel; confrontare sequenza, contenuto, metadati e overlap effettivo, conservando eventuali difetti legacy. Invariante di lunghezza verificata dopo preparazione tokenizer, senza normalizzare Markdown/LaTeX. |

Non sono emersi blocchi di correttezza o requisito. GPT-P001 amplia la copertura
osservabile e non rende invalida la prova minima delle fasi già prevista.

## Motivazione dell'esito e limiti

**GO:** il piano è circoscritto alla migrazione autorizzata, corregge packaging e
avvio con una transizione flat minima e associa tutti i criteri a prove concrete.
Le incertezze essenziali sono identificate e hanno un percorso di verifica o ritorno
al supervisore prima del GO finale. Non scambia procedure future per risultati,
build/import/mock per conversione reale, né health per disponibilità del motore.

Il GO vale per r001 sullo snapshot `plan-r001` con l'identità sopra indicata.
Non certifica lock, installabilità, Docker, fedeltà degli output o compatibilità
d'inferenza: tali risultati appartengono all'implementazione e alle sue due review.
V7/V8 restano essenziali; V10/V11 rinviate comportano il limite esplicito previsto,
senza affermare supporto Marker/GPU collaudato. Gli accessi primari non riusciti
sono registrati nelle evidenze e non sono usati per attestare dati non letti.

Consegna al **supervisore**, che attende entrambi i report validi sul medesimo
snapshot, arbitra GPT-P001 e gli eventuali rilievi dell'altro revisore e solo dopo
un arbitrato GO prepara il prompt della chat implementatrice. Nessun registro
condiviso, piano, snapshot, codice o output del concorrente è stato modificato.
