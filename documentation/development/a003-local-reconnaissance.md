# A003 — Ricognizione locale prima del piano

2026-10-12 Europe/Rome. Letture statiche/stdlib, comandi informativi Pandoc e
metadati Docker: nessun import ML, installazione, download di modelli,
conversione o container avviato. Fonti pubbliche consultate solo per capacità
e licenze dichiarate. Tutte le prestazioni a003 restano **NOT_MEASURED**.

## Hardware osservato

Intel i7-1165G7, otto CPU logiche e otto nell'affinità osservata. `/proc/meminfo`
riporta 65.581.248 kB totali (circa 62,54 GiB) e circa 37,31 GiB disponibili
alla lettura; filesystem del repository con circa 132,09 GiB liberi. Sono
fotografie dell'host, non quote o garanzie di disponibilità per le prove.

Dispositivo display PCI Intel `0x8086/0x9a49`; `nvidia-smi` e nodi `/dev/nvidia*`
assenti nell'ambiente osservato. Nessun profilo CUDA misurato. I file cgroup v2
standard cercati non erano presenti: limite effettivo del processo ancora da
accertare. Nessuna certificazione GPU, WSL2, Mac/MPS o altro host.

## Capacità dichiarate, artefatti e misure

| Candidato | Capacità documentata | Disponibilità osservata | Misura a003 |
| --- | --- | --- | --- |
| Pandoc | Conversione nativa e AST JSON; licenza GPL | `/usr/bin/pandoc` 3.1.3; elenco reader include DOCX/EPUB/RTF/LaTeX, non PDF | NOT_MEASURED |
| Marker pin del progetto | README v1.10.2: PDF/immagini, formule, tabelle, asset, CPU/GPU/MPS; LLM opzionale | Wheel 1.10.2 e Surya 0.17.1 in cache a001; immagine CPU presente; assente nella venv principale | NOT_MEASURED; inferenza e completezza pesi non verificate |
| Docling | README corrente: PDF, ordine, tabelle, OCR, export Markdown/JSON e uso locale | Assente tra i package selezionati della venv principale; cache Docling/Hugging Face standard assenti | NOT_MEASURED; versione/profilo da pianificare |
| Percorsi diretti/legacy | Script e output reali a002 già conservati | Package principale 1.0.0 nella venv; canonici/raw/gold presenti | Nessuna nuova prova; risultati a002 ricevuti con i propri limiti |

Fonti primarie consultate: [Pandoc](https://pandoc.org/),
[Marker v1.10.2](https://raw.githubusercontent.com/datalab-to/marker/v1.10.2/README.md),
[Docling](https://github.com/docling-project/docling).
Non attribuire dichiarazioni upstream alle versioni installate senza verifica.

La venv principale risolve al managed locale CPython 3.12.13 sotto
`.venv-python`; è osservazione statica di symlink/config, non preflight di origine
o attestazione S/B/I corrente. Gli ambienti root `.venv-marker`,
`.venv-marker-gpu` e `venv` non esistono. La ricognizione non inventaria tutte
le venv storiche dentro temp né runtime globali; eventuali altri percorsi
devono essere identificati prima dell'uso. La sola presenza di uv, Docker,
pdftotext, pdftoppm e Tesseract non ne verifica profili o lingue OCR.

## Cache, immagine e licenze per versione

Cache `/home/davide/.cache/datalab/models`: cinque famiglie, 40 file,
2.222.300.876 byte osservati (circa 2,07 GiB). Famiglie:
text_recognition/2025_08_29, text_detection/2025_05_07,
table_recognition/2025_02_18, ocr_error_detection/2025_02_18 e layout/2025_02_18.
Nessun peso letto o caricato, hash dei grandi file non calcolati. Il model card
OCR locale è un template con licenza non specificata: non prova provenienza,
compatibilità, diritti o sufficienza dell'intero insieme dei pesi. L'eventuale
riuso richiede manifest e licenze puntuali e assenza di download impliciti.

Immagine `run-a001-marker-cpu:r007`, ID
`sha256:b3db6dc22d92d7771aba40013bab4281ebe9f68188d5c46fc2617136b7d2c3a8`,
1.396.495.099 byte, RepoDigests vuoto. Letti soltanto elenco e inspect ID/size;
nessuna ispezione dinamica dei package o inferenza. Il contratto Docker a001
non equivale a conversione PDF riuscita. Il primo elenco ha restituito
permission denied sul socket; escalation dello strumento accettata per la
sola lettura. Fallimento conservato, nessun rifiuto di approvazione né modifica
dei privilegi persistenti. Le immagini estranee al progetto non sono input.

Metadati delle wheel locali: Marker 1.10.2 e Surya 0.17.1 dichiarano
GPL-3.0-or-later; torch 2.7.1+cpu BSD-3-Clause. Sono archivi disponibili,
non prova dell'installazione attuale nell'immagine. Il README Marker del
[tag v1.10.2](https://raw.githubusercontent.com/datalab-to/marker/v1.10.2/README.md)
distingue GPL del codice e licenza modificata dei pesi. L'[upstream corrente](https://github.com/datalab-to/marker)
dichiara Apache-2.0 per il codice e un diverso percorso di inferenza; non
trasferire licenza, requisiti o prestazioni al pin locale, né aggiornare il
progetto implicitamente. Versione nuova significa profilo e preparazione nuovi.
[Docling](https://github.com/docling-project/docling#license) dichiara MIT per
il codice e rimanda alle licenze dei singoli modelli; non ne segue una licenza
unica dei pesi. Nessuna decisione distributiva o commerciale presa qui.

## Evidenze e limiti

Evidenze locali nella run `temp/run-a003-fase1-engine-comparison/evidence/`:
`reconnaissance-local-r001.json`, `reconnaissance-archives-r001.json`,
`docker-image-list-r001.json` (fallimento), `docker-reconnaissance-r002.json`
(lettura riuscita), `administrative-reception-r001.json` e
`selected-inputs-r001.json`. Questi file attribuiscono osservazioni e hash
selezionati; non sono benchmark o snapshot di inferenza. Costo del singolo
probe stdlib registrato dove disponibile; amministrazione/startup/tool/browser
complessivi non misurati, nessun totale CPU esaustivo dichiarato.

La base `dev`/HEAD e il ref locale `origin/dev` sono a `9f702a2`; commit
amministrativo con padre 7caff33 e 24 log archiviati verificati, senza query
remota o prova di push. [Mandato e stima di risorse](a003-engine-comparison-brief.md):
branch manuale e piano restano i prossimi passi. Nessuna nuova quota disposta.
