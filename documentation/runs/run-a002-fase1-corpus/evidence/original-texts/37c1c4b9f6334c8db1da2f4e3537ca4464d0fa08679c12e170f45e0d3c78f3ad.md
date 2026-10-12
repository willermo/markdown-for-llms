# Review indipendente piano a002 r001 — ChatGPT

Agisci esclusivamente come **reviewer ChatGPT** per la run a002. Questa chat
revisiona il piano come ruolo separato da autore e supervisore; registra il
provider effettivo e segnala eventuali differenze senza fingere una sostituzione.
Non leggere la review concorrente o l'arbitrato corrente e non modificarne i file.

Output esclusivamente tuoi:

- Report: `temp/run-a002-fase1-corpus/reviews/review-plan-r001-chatgpt.md`.
- Evidenze: `temp/run-a002-fase1-corpus/evidence/review-plan-r001-chatgpt/`.
- Checkpoint: `temp/run-a002-fase1-corpus/handovers/review-plan-r001-chatgpt.md`.
- ID rilievi: `GPT-P001`, `GPT-P002`, progressivi.
- Prompt di origine: `temp/run-a002-fase1-corpus/prompts/02-review-plan-r001-chatgpt.md`.

## Oggetto comune e identità

Run `run-a002-fase1-corpus`, review iniziale del **piano r001**.
Stato PLAN_REVIEW, nessuna implementazione autorizzata. Il supervisore ha
ricevuto e identificato la consegna; ricezione e snapshot non sono un GO.

- Piano: `temp/run-a002-fase1-corpus/plans/plan-r001.md`.
- SHA256 piano: `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`.
- Snapshot comune: `temp/run-a002-fase1-corpus/snapshots/plan-r001.json`.
- SHA256 del file snapshot: `db1dfa3a54fb7d7235e96030a196ba546409c7cd4ea18ceda3753b80edb16026`.
- Impronta working tree: `2aef47fc0f089f82c9d6904cef67d5aa04da3de3232eebb055642021c0ea79d8`.
- Branch: `feature/run-a002-corpus`; HEAD/dev/merge-base:
  `80c31da46e5caab990d3596c79c44e524ec1dc95`.
- Snapshot: 132 file di worktree e 9 artefatti espliciti.

Sono inclusi piano, checkpoint del pianificatore, prompt01, brief, quattro
JSON di evidenza planning (incluse entrambe le verifiche documentali) e
ricezione del supervisore. `temp/` e `documentation/runs/` sono esclusi
dall'impronta generale; gli artefatti espliciti sopra sono invece vincolati.
I checkpoint vivi e i report nuovi dei reviewer sono output esclusi, quindi
non alterano lo snapshot. L'esclusione dell'archivio non permette di nascondere
codice prodotto o ignorare prove pertinenti.

Il branch è ora presente: la precedente rilevazione di assenza nel prompt01 e
nei checkpoint storici è riconciliata in STATE. Base immutata, indice vuoto,
delta documentali locali attesi; non usare soltanto `git diff base...HEAD`
per cercare lavoro non committato. Nessuna query remota nuova del supervisore.

## Letture e verifica iniziale

Usa le skill `.agents/skills/manage-implementation-run/SKILL.md` e
`.agents/skills/verify-conversion-fidelity/SKILL.md` e il protocollo della run.
Leggi nell'ordine, evitando scansioni indiscriminate di temp o della storia a001:

1. `AGENTS.md`, `temp/HANDOVER.md`, sezione corrente iniziale di
   `temp/PROJECT-CONTEXT.md`, STATE/HANDOVER/brief della run e questo prompt.
2. `documentation/development/run-lifecycle.md`, le due skill indicate,
   `documentation/development/templates/review.md` e il checkpoint corrente
   `temp/run-a002-fase1-corpus/handovers/supervisor-plan-review-r001.md`.
3. Piano r001 **integrale**, prompt01 e `handovers/planning-r001.md` della run;
   i quattro JSON sotto `evidence/planning-r001/`, la ricezione e la receipt
   snapshot sotto `evidence/supervisor-plan-review-r001/`.
4. `documentation/README.md`, registro ADR, ADR0001/0004, fase1 roadmap,
   `documentation/development/next-run-a002-corpus.md`, open-questions e
   `.agents/workflows/converter-benchmark.md`.
5. `docs/how-to/ambiente-uv.md`, `docs/reference/toolchain-legacy.md`,
   `pyproject.toml`, `MANIFEST.in`, parti pertinenti di `unified_converter.py`,
   `tests/conftest.py` e `scripts/diagnostics/run-a001-fase0-uv/` (R, preflight,
   S/B/I, launcher/tests, input inventories). Leggi il codice necessario a
   verificare fattibilità/contratti, senza importarlo o eseguirlo implicitamente.
6. Solo se pertinente al rilievo: manifest a001 e note di materiale/pulizia
   già ricevuta. A001 GO e limiti NON PASS non sono prove del nuovo codice.

Verifica branch/HEAD/dev, merge-base, indice, modificati e nuovi non ignorati,
hash di piano e snapshot. Verifica lo snapshot **prima e dopo** la review:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label plan-r001
```

Questo uso stdlib del gestore di contesto non è una prova del prodotto Python.
Registra argv/cwd/output/exit delle verifiche nelle evidenze del tuo ruolo.
Non rigenerare lo snapshot né correggere gli input. Se è STALE o l'identità non
corrisponde, conserva il risultato e segnala al supervisore l'invalidità
dell'oggetto ricevuto; non dichiarare valida una review del precedente stato.

## Criteri tecnici comuni ai due reviewer

Valuta la conformità al brief e al mandato completo, la coerenza interna e la
fattibilità delle prove proposte. È review del **piano**, non del codice futuro:
fixture, runner, gold umani e benchmark non sono ancora stati implementati.
Non attribuire PASS runtime alle sonde di versione/help; non chiedere prove
non ancora dovute per il solo fatto che il piano le prevede. Un prerequisito
futuro esplicito si valuta per assegnabilità/costo e per il criterio di chiusura;
non si scambia con una verifica già svolta o con una deroga automaticamente accolta.

| Criterio | Domande e verifiche sostanziali |
| --- | --- |
| C01 obiettivo e confini | Il risultato completo copre corpus, gold indipendenti, protocollo/runner, positivi/negativi e baseline reale? Contratto benchmark minimo distinto dal dominio di fase2, nessun ML/remoto/modello o Git implicito? Il lavoro necessario resta entro l'incremento, senza apparato sproporzionato? |
| C02 corpus e provenienza | I 10 ID e 9 pagine PDF coprono formati/IT-EN/colonne/scansioni/misto anche nella stessa pagina, formule/matrici/numeri/celle/figure/note/bibliografia? Specifiche, asset e generatori sufficienti e indipendenti dal motore? Licenze sintetici/font, hash e manifest offline verificabili? Byte-identical e content-reproducible distinti con prova adeguata? |
| C03 gold e dipendenza umana | Trascrizioni/annotazioni confrontabili sul sorgente/render effettivo, controllo completo di overflow/glifi/pagine/celle/ordine? Gold non derivato dal motore o dal comparatore? L03 impone 60–90 min umani non ancora assegnati: il requisito è motivato, sostenibile e completabile con una sessione reale e lavoro indipendente? Nessuna firma umana agente, nessun VERIFIED inventato? Evidenziare cosa deve essere disposto prima/durante il mandato, senza impersonare il verificatore. |
| C04 reader e allineamento | Osservazioni gold-blind, offset/raw conservati, parser limitato dichiarato e unknown visibili? Matching one-to-one, duplicati/aggiunte e blocchi mancanti inclusi, inversioni non ridotte a omissioni? Euristiche 0,75/0,15 motivate e calibrabili senza alterare soglie di fedeltà? Gestione ripetuti/ambiguità non assegna PASS spurio? |
| C05 metriche e matematica | Unità/denominatori/normalizzazioni per testo Unicode, numeri/segni, pagine/blocchi/ordine, formule/righe/matrici, celle vuote, asset/caption e riferimenti? CER/WER o similarità non occultano perdite critiche? Tipografia/sintassi/semantica distinte; gate manuali/unavailable non trasformati in PASS? Soglie esatte motivate per sintetici senza pretesa universale OCR? |
| C06 positivi e negativi | Positivi discriminano un comparatore che rifiuta tutto? Mutazioni esercitano page/block loss, numero/segno/simbolo, ordine, cella assente/vuota, asset/caption, note/link, raw vuoto e batch con errore? Assert su contenuto, denominator, localizzazione, stato ed exit, non sola creazione file? Baseline/gold canonici preservati? |
| C07 identità e stati | Document/experiment/attempt/config/hash identificano omonimia e riesecuzione? Raw/errori/parziali conservati; processo exit0 non basta per completezza/fedeltà? Stati conversione/bundle/qualità/batch/coverage e precedenze exit coerenti? Aggregati non eliminano FAIL, unavailable o denominatori attesi? |
| C08 baseline reale e lacune | Cinque percorsi diretti TXT/Pandoc e tre legacy installati realmente praticabili? Pandoc media/resource/cwd/writer e expander TeX compatibili con tool/routing effettivi? Nessuna riparazione del motore usando gold/asset attesi? ≥1 nativo completo con asset come richiesto, FAIL di qualità pubblicabili e batch fallito visibile? L01 PDF/OCR e L02 compilazione TeX restano lacune precise, senza dichiarare successo del confronto motori? |
| C09 contenuti non fidati | Confini path/symlink/hardlink/ZIP/XML/HTML/SVG/includes/font/resource coerenti col corpus e tool disponibili? Niente comandi da dati, fetch remoti o fallback impliciti; shell=False e registry esplicito? Limiti tecnici concretamente attuabili, negativi prima di scritture o esecuzioni pericolose? R protegge rete ma non viene spacciato per sandbox filesystem universale? |
| C10 toolchain e prove | Nuovi input managed 3.12.13/wheel/S/B/I/dieci hash e R nello stesso interprete dei figli, preflight prima degli import/no-sync? Runner/fixture/helper esterni alla wheel hanno manifest/hash/audit prima-dopo, test tampering e provenance reale? Snapshot esecuzione autonomi, output vivi esclusi, solo prove dipendenti ripetute? Suite selezionata pertinente senza nuove campagne a001/API/Docker/ML implicite? |
| C11 risorse e recupero | Quota proposta 15 h wall/7200 CPU-s/4 GiB nuovi/RAM operativa 8 GiB/rete 0 è credibile rispetto a implementazione/render/test/32 attempt/verifica umana/fix/registri? 10 h base+3 h margine+2 h riserva; CPU/disco/log separati e misure RSS/wait4/figli con limiti espliciti? Stop con riserva attuabile, contatori senza reset, deadline ≤900 del tool e log/file caps compatibili? Non confondere cap, stima e upper misurato; niente download per rimediare collection o dipendenza offline assente. |
| C12 consegna e ciclo | A01–A12 hanno prova/esito/costo e dipendenze; L01–L05 motivate senza indebolire requisiti? Guide/report creati su prove effettive con stati chiari, archivio e checkpoint utilizzabili? Errori ordinari/freeze/argv/permessi correggibili nella stessa chat, escalation tool mantiene isolamento e rispetta un rifiuto effettivo? Ruoli/doppia review/arbitrato e Git manuale preservati? |

Valuta il piano nel suo insieme: una scelta reversibile non enumerata può
completare il requisito entro il mandato; non trasformare ogni dettaglio di
implementazione in un blocco. Viceversa non ignorare una contraddizione o una
prova essenziale irrealizzabile perché viene chiamata dettaglio operativo.
Il FAIL aritmetico planning r001 è preservato e corretto nella r002; verifica
la versione corrente, non attribuirle automaticamente il difetto storico.

Per ogni rilievo collega posizione precisa (sezione/riga), percorso/input ammesso,
impatto, requisito violato e attribuzione: introdotto dal piano, ereditato
legacy, causato dal mandato o dipendenza da disporre. Un difetto preesistente
può bloccare un requisito esplicito se raggiungibile nel percorso previsto;
una miglioria fuori scope resta suggerimento motivato. Proponi correzione
minima e prova di chiusura. Non inventare rilievi per riempire una tabella.

## Autonomia del reviewer e limiti

Consentiti letture/ispezioni statiche, verifiche hash/snapshot/link/aritmetica,
sonde di help già pertinenti se necessarie, scrittura del proprio report,
evidenze e checkpoint. Niente implementazione, mutazioni di piano/fixture/
prodotto, installazioni, build, benchmark, prove R, import ML o acquisizioni.
Se una questione richiede una prova futura, identifica la prova nel rilievo,
non avviare la fase implementativa. Eventuale consultazione di versioni/licenze/
capacità candidate deve usare fonti primarie aggiornate con URL/data; la review
può restare sulle versioni locali documentate senza un nuovo sondaggio motori.

Non modificare STATE/HANDOVER/eventi/indici/changelog/checkpoint del supervisore,
file dell'altro reviewer o snapshot. Non avviare agenti/reviewer delegati;
questa è una delle due nuove chat indipendenti richieste dall'utente.
A001 è integrata con GO e limiti NON PASS; pulizia già ricevuta, non ripeterla.

## Report ed esito

Usa il template review. Dichiara autore/provider/modello effettivo e chat ID
se disponibile, prompt di origine, piano/r001/hash, snapshot/hash/impronta,
branch/HEAD/base/delta, indipendenza e verify prima/dopo. Se l'identificativo
esatto del modello non è esposto, dichiaralo senza inventarlo.

Esito **GO** oppure **NO_GO** motivato sul piano identificato. Distingui blocchi
reali da suggerimenti e prerequisiti espliciti da disporre; un GO non è prova
che il runtime futuro sia già passato. Se la review non può essere conclusa
sull'oggetto corretto, dichiara invalidità/impedimento fattuale e prossimo passo,
non produrre GO dalla sola assenza di report o da snapshot stale.

Per ogni rilievo includi ID, severità/blocco sì-no, posizione, raggiungibilità,
attribuzione/requisito, problema/impatti, evidenza e criterio di risoluzione.
Riporta tutti i criteri effettivamente controllati e i limiti delle prove;
non leggere l'altro report o l'arbitrato corrente prima della consegna.
Nel checkpoint indica input, output, esito, processi propri pendenti e
prossimo ruolo **supervisore per ricezione di entrambi i report e arbitrato**.
Concludi in chat con esito reale e percorsi. Non avviare implementazione.
