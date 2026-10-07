# Implementazione r001 — promozione lock e input reali delle prove prodotto

Agisci come **implementatore** in `/home/davide/workarea/markdown-for-llms`,
stessa chat ammessa. Esegui la promozione ricevuta e acquisisci gli input runtime
base necessari a S/B/I/E. Delega per obiettivo r013: scegli e correggi mezzi
reversibili entro requisiti/costi; nessun handoff per un errore ordinario.

Leggi AGENTS, skill manage-implementation-run, protocollo, indici/roadmap,
STATE/HANDOVER e checkpoint autore/report-r011; poi r014 e checkpoint supervisore
`temp/run-a001-fase0-uv/handovers/supervisor-product-inputs-r001.md`.
Input di questa consegna, relativi alla run:
- `evidence/supervisor-implementation-r001/product-input-reception-r001/`:
  reception.json, authorized-scope.json, transition.json;
- `evidence/implementation-r001/resume-package-s015/next-product-gate-request-r001.json`,
  lock-audit.json, ricevute lock/check e final-checks;
- piano/arbitrato r003 (P2/D1–D5 e criteri), r013 e r014 correnti.

PianoSHA462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b,
arbitratoSHAf14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d
immutati; GO piano soltanto, NO_GO storici conservati. Scope 63176byte,
SHA3fd09c86e4cd4d124c3df035cf84ee102b97fee7c372a5e27689d8463134e4c7; r014 SHA2ee454ee9820745ba0c1d354528410811ec2d04ad319304d10d89473c0b4e0ff.
Leggi lo scope completo per argv/env/cwd/costi, ricalcola le identità.
Branch feature/run-a001-uv HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto; modifiche core/documentali preesistenti identificate.

## Consegna già ricevuta e contesto

Lock e check nella copia PASS, nuovi R/D4/wrapper, lock287148byte/SHA
b3288f1d51b880d9c683a43607d9eedc5b8c87465c8a1d357818c6b61d12f05f,
143package/141nomi, fonti CPU/cu126/EbookLib invariati. Questo non è prodotto
installato o S/B/I/E PASS. Le identità di input e ricevute in scope sono protette.
S015 MATCH alla ricezione, ora **storico** per cinque delta documentali dichiarati
in transition e per le prossime modifiche autorizzate. Non esigere MATCH del
vecchio worktree con nuovi input; conserva lo snapshot immutato e verifica
soltanto i suoi input pertinenti non modificati. Nessun snapshot nuovo esiste
ancora; `impl-r001-stage-product-s016` è label **riservata**, non prova pronta.

## Esegui il lavoro prima del freeze

1. Verifica origine/binari uv0.10.10 e managed locale CPython3.12.13, startup,
   config ed environment chiusi dello scope. Root uv.lock/.venv, work
   `work/product-sbi-r001` e `/tmp/run-a001-product-wheel-env-r001` devono essere
   assenti all'ingresso. Non avviare nuovi interpreti prima della guardia di
   origine/config/site pertinente. Non ricontrollare l'intera baseline storica
   senza un delta o anomalia; conserva le sue identità e ricevute.
2. Preserva ingresso root pyproject. Dalla copia già ricevuta porta **soltanto**
   tool.uv.dependency-metadata EbookLib0.18/requires-dist lxml/six, oppure copia
   il pyproject identico dopo confronto TOML che dimostri quel solo delta.
   Pyproject risultato atteso2737byte/SHA
   60ef6f17da3f6787b14137749c5143323117dd2e9b257748cc556def5060b856.
   Copia byte identici del lock nella radice, non rigenerarlo. Verifica coppia/
   origini/pin/extra/markers/indici/conflicts immutati. Nessun commit/staging.
3. Nuove evidenze `evidence/implementation-r001/product-inputs-r001/`, work/
   tmp/config esclusivi dello scope. Usa **runtime-acquisition** per uv sync
   --locked --no-default-groups --no-install-project --no-editable --no-build,
   managed assoluto e stesso cache; argv/cwd/env completi in prefix_operation.
   Nessun extra o dev/API/ML, nessun backend o install del progetto. Rete pubblica
   per l'acquisizione base; non dichiarare R offline sul download. Argv strutturati,
   shell=False/close_fds=True/env puro senza merge os.environ, no verbose/quiet.
   Guardie pre/post, comandi/exit/provenienza/hashes e costi reali.
4. Inventaria il runtime base realmente selezionato dal lock, origine/interprete,
   artefatti backend setuptools84 già acquisiti e build inputs reali. Se uv
   conserva cache espansa anziché archivi whl, identifica i file reali, RECORD
   e digest upstream distinti dai byte osservati. Se servono wheel d'ingresso,
   acquisisci URL/hash esatti del lock con letture HTTPS limitate nel budget;
   nessun resolver/installer sostitutivo, nessun hash di un futuro download.
   Costruisci `work/product-sbi-r001/package-inputs.json` nel formato del produttore
   S (schema1/scope package-inputs/files/backend), senza output futuri S/B/I/E.
5. Completa e verifica staticamente i template della sequenza ricevuta in scope:
   R/D4, S, build offline da sdist con setuptools84 locale/constraints84, B,
   root/base-a/base-b sync offline+reinstall wheel canonica, I/E, venv esterna
   con runtime export hash-locked e canonica. R/D4 deve restare sul medesimo
   comando/interprete dei figli. Non affermare backend/boot attestati da controlli
   post: predisponi attestazione reale pre-build e traccia del backend.
   Nomi/digest di sdist/wheel derivano da B effettivo; qui sono attese.
   Sono delegati reader/launcher/template e fix locali dei diagnostici uv
   necessari prima freeze, mantenendo i criteri e documentando prove invalidate.
   Non cambiare dieci moduli del prodotto, baseline, fonti/pin o requisiti.
6. Consegna request **WAITING_FOR_STAGE_SNAPSHOT**, owner supervisor, label
   `impl-r001-stage-product-s016` in `implementation/stages/` con request.json,
   piano/arbitrato/r014/scope, lista esatta di input esistenti senza symlink/
   path assoluti/escape, inventario/copie/fixture pertinenti, ambiente/comandi
   effettivi, residui costi/tempi. Usa i path reali dei file canonici, non link
   di venv come artefatti. S include moduli/build/diagnostics/inventory: assicurati
   che la lista li copra tutti. Nessuna inclusione di S/B/I/E futuri.

**Non eseguire S ufficiale, build/install progetto o prove successive prima
che il supervisore congeli questi input.** Non auto-delegare snapshot. Questa
è la dipendenza D1 input→snapshot→S, non un nuovo piano o autorizzazione per
ogni comando. L'intero seguito S/B/I/E base è già ricevuto condizionatamente
al freeze; consegnalo pronto a una ripresa diretta con stessi costi, non una
nuova proposta astratta. Non costruire S standalone e dichiararlo ufficiale.

## Costo ricevuto per tutta la fase, senza reset

Pool distinto **112MiB** da Hentry=553541632, tutto il pregresso incluso
nel totale. Vecchio residuo32MiB non aggiunto; prefix e prove successive
consumano lo stesso112MiB, nessuna nuova tranche ad ogni label. Ledger run,
.venv-python, .venv e /tmp/run-a001-product-wheel-env-r001 con lstat, directory/
link inclusi senza seguirli. H=max(logical,allocated), Delta=max(0,H-Hentry),
Delta<117440512 e H+max(0,117440512-Delta)+16777216<939524096. Pool1GiB,
stop896/libero1GiB repository/riserva esterna16MiB; /tmp separato deve avere
almeno128MiB liberi (112fase+16riserva). Controlla entrambi periodicamente.
Stime8MiBwheel/64env/24backend/8registri+8contingenza sono stime, non cap separati.
Monitor0,5s/gap target1s non quota atomica; file32MiB/JSON8MiB/stream1MiB.
Workload complessivo7200s per prefix+seguito; singolo fino900s entro residui,
outer=figlio+180; raccogli sessioni/processi, retry solo dopo diagnosi.

Fonti configurate/endpoint verificati e wheel del solo runtime base coperti.
No download Python/backend upgrade/extra pesanti/pesi/font/MLpayload/paid,
no import applicativo/entrypoint/suite/V0/ABI impliciti. Startup/origine/site
vanno verificati; fixture/config/namespace/daemon/private input protetti.
Nessuna modifica host/privilegi/socket/profili/rete, no cleanup/reset/spostamento
fuori ledger. Decisioni reversibili non enumerate si prendono nella stessa
chat; escalation per costo/requisito/fonte logica nuovi, privacy/confinamento
non rispettabili, input confronto alterati, rifiuto sandbox/irreversibilità.

Scrivi delivery nuovo e `implementation/report-r012.md`, checkpoint autore con
ingresso preservato, request pronta e ricevute reali. Se i prerequisiti pronti:
**WAITING_FOR_STAGE_SNAPSHOT**, prossimo supervisore con **prompt47** nella
nuova chat; se vero blocco, WAITING_FOR_SUPERVISOR_RECEPTION con motivo concreto.
Non modificare stato comune/changelog/eventi/snapshot. No Git/deploy, nessun
servizio automatico da attendere. V1–V9/dev/API/suite e comparativi futuri,
V7/V8 costi pesanti separati/V10V11 esclusi, due review reali e GO finale aperti.
