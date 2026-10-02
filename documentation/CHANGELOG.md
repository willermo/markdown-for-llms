# Changelog dello sviluppo

Registro degli avanzamenti della nuova versione del progetto, in ordine cronologico
inverso. Distingue decisioni approvate, lavoro realizzato, verifiche e integrazione
Git; la presenza di una voce non equivale a un rilascio.

La [roadmap](roadmap.md) descrive il percorso previsto, gli
[ADR](decisions/README.md) conservano le motivazioni delle decisioni e l'
[archivio delle run](runs/README.md) conserva le evidenze permanenti. Il contesto
operativo corrente rimane in `temp/`, esclusa da Git.

## Stato al 2026-10-02

| Area / fase | Stato verificato | Prossimo passo |
| --- | --- | --- |
| Fase 0 — Base documentale e governance | Struttura iniziale committata in `f945c4b`; estensione del ciclo delle run predisposta nel working tree | Completare commit e integrazione manuale del bootstrap in `dev` |
| Fase 0.1 — Migrazione a uv | Approvata; run `run-a001-fase0-uv` in `PREPARATION`; nessun piano revisionato o GO | Creare il feature branch da `dev` aggiornato, pianificare e svolgere le due review |
| Fasi 1–8 — Benchmark e nuova applicazione | Da iniziare | Seguire dipendenze e criteri della roadmap |
| Codice applicativo | Pipeline legacy; nuova architettura non implementata | Migrazione della toolchain prima dei benchmark |
| Integrazione e rilascio | Lavoro su `feature/document-converter-v2`; nessun merge del bootstrap in `dev` o promozione a `main` registrato | Operazioni Git manuali dell'utente |

## Non rilasciato

### 2026-10-02 — Ciclo delle run, approvazione uv e registro degli avanzamenti

**Fase:** estensione del bootstrap e preparazione della fase 0.1.
**Stato:** governance e strumenti predisposti localmente, da committare e integrare.

#### Realizzato

- Creato il [protocollo delle run supervisionate](development/run-lifecycle.md):
  pianificazione, doppia review indipendente, arbitrato, implementazione, doppia
  review e arbitrato finale, con ritorno alla pianificazione in caso di `NO_GO`.
- Aggiunta la [skill locale del ciclo di implementazione](../.agents/skills/manage-implementation-run/SKILL.md)
  e aggiornati `AGENTS.md` e i workflow per rispettare ruoli e integrazione Git manuale.
- Predisposti template per stato, piani, review, arbitrati, report, handover e archivio.
- Creata `temp/`, ignorata da Git, con contesto globale, run `run-a001-fase0-uv`
  e prompt per una nuova chat supervisore.
- Introdotto [run_context.py](../scripts/run_context.py) per inizializzazione dei
  contesti e verifica delle impronte degli artefatti, con [test dedicati](../tests/governance/test_run_context.py).
- Definite destinazioni permanenti per report, evidenze, probe e test prima della
  pulizia selettiva delle run.
- Aggiunto questo changelog, collegato all'indice e alle regole di manutenzione.

#### Decisioni e pianificazione

- Approvata l'adozione di uv in [ADR 0006](decisions/0006-python-toolchain-uv.md).
- Adottato il ciclo supervisionato in [ADR 0007](decisions/0007-supervised-development-runs.md).
- Inserita la fase 0.1 nella roadmap, anticipando la migrazione della toolchain ai benchmark.

#### Verifiche ed elementi aperti

- Il bootstrap dell'helper è stato verificato con **6 test passati** eseguiti tramite
  `python3 -m unittest discover -s tests/governance -v`.
- Sono stati validati il formato delle quattro skill locali, i collegamenti
  documentali e l'esclusione di `temp/` da Git.
- La migrazione uv non è stata implementata; le review indipendenti della run non
  sono ancora iniziate. I controlli del bootstrap non costituiscono un `GO` sulla run.
- Nessun benchmark con modelli reali o conversione cloud eseguito in questo intervento.

### 2026-10-02 — Base architetturale della nuova versione

**Fase:** 0 — Base progettuale.
**Stato:** committata sul feature branch; riferimento `f945c4b`.

- Creato `feature/document-converter-v2` dal `dev` locale al commit `8fa5be9`.
- Archiviata la [proposta preliminare](architecture/preliminary-draft.md), con
  attenzione a fedeltà, formule, OCR, asset e chunking opzionale.
- Registrati gli ADR 0001–0005 su contenuti, FastAPI/Web UI, persistenza,
  configurazione/distribuzione e documentazione/governance.
- Definiti [requisiti Web UI](architecture/web-ui.md), roadmap, questioni aperte,
  [criticità legacy](legacy-findings.md) e valutazione MCP.
- Predisposti `AGENTS.md`, le prime tre skill locali e cinque workflow.
- Creata la struttura [Diátaxis in `docs/`](../docs/README.md) e aggiunti i collegamenti
  nel README principale, mantenendo riconoscibile la guida legacy.
- Verificati nella consegna iniziale tre skill, 30 documenti e 69 collegamenti locali.

## Come aggiornare il registro

- A ogni avanzamento significativo aggiungere una voce datata con fase/run, risultato,
  verifiche realmente eseguite, limiti e prossimo passo; aggiornare il quadro di stato.
- Il supervisore registra i passaggi della run e gli esiti degli arbitrati verificati.
  Gli autori aggiornano le voci relative alle modifiche effettivamente consegnate.
- Registrare commit, merge, release e deploy solo dopo averne verificato l'esito,
  aggiungendo i riferimenti reali. Non assegnare versioni di rilascio non definite.
- Conservare la cronologia: nuove voci documentano integrazioni o cambi di stato;
  correggere voci precedenti solo per inesattezze, rendendo esplicita la correzione.
- Collegare roadmap, ADR e archivio permanente senza duplicare report estesi o
  dipendere da link a file temporanei che saranno rimossi.
