# A004 — Primo prodotto locale utilizzabile

17 ottobre 2026, sintesi del supervisore. **GO tecnico I1 r004004 dopo due
review indipendenti GO/GO e arbitrato.** Stato **INTEGRATED**: merge e push
di dev verificati il 17 ottobre; nessun deploy. Primo uso/P14 PENDING.

Alla consegna: branch `feature/run-a004-usable-product`, HEAD
`f0260568aa544b36b15c351d366583ddd9f8973f`, base/dev
`9f702a2d368cb8f4a3a6bfb7b9146c6757ecd9df`.
[Passaggio Git manuale](manual-integration.md),
[guida d'avvio](../../../docs/tutorials/primo-pdf-locale.md),
[decisione ADR0009](../../decisions/0009-first-usable-product.md).

Integrazione manuale eseguita dall'utente e verificata: feature
`323f17208bb70d9fc8ee5ad3303d7874516fac5b`, merge dev
`f8bf44c86736bae503f345bcab99fff8eeee2708` (primo genitore base9f702a2).
Albero identico feature/dev: `e65a374b56d6e9aa7039d5b9eaa3649a5550f78b`.
Interrogazione diretta origin conferma entrambi i branch; main ancora
`428d404d84fa01ce4ba6f5c74c12a26bdf06d109`. 731 file e 561 artefatti del
contesto precedente invariati, 2051 percorsi del commit uguali alla selezione
approvata, incluse 1967 cancellazioni. Worktree pulito alla ricezione.
Il verify storico segnala HEAD/branch e omissione dei 13 record deleted dei
JSON ormai cancellati dall'index Git: tutti i 384 file presenti e gli
artefatti coincidono. Non sono necessarie nuove review per questo passaggio.

## Risultato e verifiche

Percorso PDF→Markdown originale/asset con Marker CPU, CLI/API/Web UI sugli
stessi servizi, archivio SQLite/filesystem e worker separato confinato senza
rete. Browser con testo inerte e galleria PNG/JPEG verificata, download
sorgente/ZIP, coda, annullamento, ripetizione e recupero di pubblicazioni
complete prima/dopo rename. Inventario/integrità e qualità sono stati distinti.
È il primo incremento delle fasi2/3, non l'intera applicazione della roadmap.

Canonica `implementation-r004-build-004`, suite `suites-005`: **3240 test e
440 subtest PASS**, zero skip (3156 legacy, 7 API, 11 packaging, 66 prodotto).
Ricevute ricontate da supervisore e reviewer, non suite rieseguita dal
supervisore. I reviewer hanno eseguito ciascuno 25 test/62 subtest mirati.
22 file package e dieci legacy concordi fra sorgente/wheel/installazione.
Browser effettivo, ZIP e prove quota/crash/recupero attribuiti alla build.
Docker, smoke radice e collaudo completo di tutte le piattaforme esclusi.

C06 reale r003 sul monitor finale: prosa, numeri, tabella e figura utili nel
campione; frazione, somma e matrice appiattite, aligned separato. R004 usa
un replay dichiarato con sorgente, raw e JPEG identici, senza nuova inferenza.
C07 storico/replay. Le prove non attestano tutti i PDF o OCR scientifico.

Sei rilievi r003 chiusi: recupero prima rename, lease morte, envelope
manifest scritto/riletto (due ID), documento malformato e lettura passiva.
NO_GO r001–r003 preservati. Tre residui bassi accolti senza riaprire un ciclo:
test UTF-8 da rafforzare, breve contesa worker.lock all'avvio, recupero dello
stesso ID perso in presenza di una voce estranea non sicura nell'archivio.
Quest'ultimo caso non è prodotto dal normale flusso provato; i byte restano
conservati, ma non si promette recupero da alterazioni arbitrarie dello storage.
Dettagli e criteri nel [debito tecnico](../../development/technical-debt.md).

## Limiti e seguito

Linux x86_64/CPU, singolo utente e loopback, €0 e documenti remoti0.
Qualità **unverified**, lettura umana P14 **PENDING**. Nessuna certificazione
semantica/asset rispetto al PDF. Il limite 450s può interrompere documenti
più lunghi; RSS aggregata campionata non è un limite hard di memoria globale.
Nessuna cancellazione selettiva dell'archivio in I1; configurazione compatibile
tramite percorso amministrativo documentato, senza modifiche manuali al DB.
Altri formati, rendering scientifico, revisione, GPU/cloud e distribuzione
restano incrementi successivi. Debito avanzato a002/a003/parser/trace aperto,
con consolidamento appena tempo e risorse lo consentiranno.

## Attribuzione e disponibilità delle prove

I riferimenti sotto sono percorsi operativi locali in temp, esclusi da Git;
non sono una promessa di disponibilità dopo pulizia o su un altro clone.
Le prove selezionate qui citate sono presenti e verificate dopo la
[pulizia autorizzata](../../development/cleanup-2026-10-17.md). Rimossi i
payload superati; preservati primo uso, integrazione e dipendenze attive.

| Oggetto | Identità / percorso locale |
| --- | --- |
| Consegna e arbitrato | `temp/run-a004-fase2-prodotto-utilizzabile/reports/implementation-r004.md`; `arbitrations/arbitration-implementation-r004-r001.md` nella stessa run |
| Review ChatGPT | `reviews/review-implementation-r004-chatgpt.md`, SHA256 `f2ec88774a43bfbebc48a1c0232ba1c7166eccc257e38ae8967743b8da67395b` |
| Review Claude | `reviews/review-implementation-r004-claude.md`, SHA256 `27fd25b48f311776eabf3f267c8f6c3fb8d9b5646380c07b879dae710fd2323e` |
| Comune revisionato | `implementation-r004-review-r001`; indice SHA256 `68e2da19ccfd7a67970b18e05f3c85ae57ebce3f39caf8c338a93e0d52576e5f`. Ricezione prima dei registri:650 file/1 blob MATCH |
| Contesto approvato | `snapshots/implementation-r004-cleanup-r001.json` e `evidence/supervisor-precommit-cleanup-r001/integration-inputs-r001.json` nella run; distingue aggiornamenti editoriali dal codice invariato |
| Prove canoniche | `evidence/implementation-r004/`: canonica004, suites005, matrix003, poll001, final002, replay003 e browser002 |
| C06 ML r003 | `evidence/implementation-r003/c06-real-002/c06.zip`, SHA256 `f346a53597cb22f1aef0fa515fb0a508460e104ee539265f7e51dfd4a1cd78e3` |

Questa sintesi e le istruzioni Git sono gli unici nuovi documenti d'archivio
selezionati: nessun albero evidence, ZIP, modello, DB o copia del runtime qui.
Link locali e coerenza dei registri verificati con git diff --check prima
della consegna. Ricevute locali d'integrazione e push in
`evidence/supervisor-integration-reception-r001/` della run; nessuna nuova
suite/build/ML. Gli aggiornamenti editoriali di ricezione restano da
committare separatamente dall'integrazione applicativa già verificata.
