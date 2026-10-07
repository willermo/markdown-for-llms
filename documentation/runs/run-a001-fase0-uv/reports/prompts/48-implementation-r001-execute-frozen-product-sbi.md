# Implementazione48 — prove prodotto S/B/I/E dopo freeze product-s016

Agisci esclusivamente come **implementatore** nel repository
`/home/davide/workarea/markdown-for-llms`, stessa chat46 ammessa. Completa
l'intera sequenza base S/B/I/E già ricevuta da r014, ora con input reali
congelati dal supervisore47. Questo mandato comprende esecuzione, diagnosi,
fix locali e verifiche nel perimetro; non è una nuova sola preparazione.
Non avviare altri agenti, review finali o fasi della roadmap.

## Ingresso e identità

Leggi AGENTS, skill manage-implementation-run e protocollo, STATE/HANDOVER,
checkpoint autore `handovers/implementation-r001.md`, report-r012 e request
`implementation/stages/impl-r001-stage-product-s016/request.json`.
Poi checkpoint supervisore `handovers/supervisor-product-inputs-r001.md`,
`evidence/supervisor-implementation-r001/product-freeze-reception-r001/response.json`
e `final-checks.json`. Tutti questi path sono relativi a
`temp/run-a001-fase0-uv/`. Leggi piano r003 completo in una chat fresca;
arbitrato r003 D1–D5, addenda r004/r013/r014 e scope corrente ricevuto.

Piano SHA462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b;
arbitrato SHAf14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d.
Scope `evidence/supervisor-implementation-r001/product-input-reception-r001/authorized-scope.json`,
63176byte/SHA3fd09c86e4cd4d124c3df035cf84ee102b97fee7c372a5e27689d8463134e4c7.
R014 SHA2ee454ee9820745ba0c1d354528410811ec2d04ad319304d10d89473c0b4e0ff.
R013 SHAea0b1078513171154ea23277bf5ca4f90bd9a0e0464ed20642de9dd619baa19b.
GO del solo piano, NO_GO r001/r002 storici; nessun GO finale.

**Freeze effettivo** `snapshots/impl-r001-stage-product-s016.json`,282780byte,
SHA7700f444c7733e5527dbdc0b3f1046d8da459d0b9ed7b98cc7ee81b881fba594;
worktree c7e4cbb8edd7df545ca0dcd25d18bdf4417cb31506352cc3bd9a03270e4f47b8,
127file/900artefatti. MATCH verificato dal supervisore, da ripetere prima/dopo
le prove. Non auto-freeze né overwrite della label. Input S/B/I/E futuri esclusi.
Branch feature/run-a001-uv, HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto. Delta worktree core/docs autorizzati, nessun reset o Git write.

Ricezione47 PASS_INPUTS_ONLY:923binding/887artefatti autore, inventory805file,
nove wheel hash del lock/ZIP/payload e217 RECORD scratch verificati; backend84
330file. Pyproject2737byte/SHA60ef6f17da3f6787b14137749c5143323117dd2e9b257748cc556def5060b856,
lock287148byte/SHAb3288f1d51b880d9c683a43607d9eedc5b8c87465c8a1d357818c6b61d12f05f,
esatti della copia ricevuta; unico delta metadata EbookLib0.18/lxml/six.
Dieci moduli invariati. Questa ricezione non è S/B/I/E/R o collaudo ABI.

## Esegui la sequenza completa

Input reali `work/product-sbi-r001/package-inputs.json`,240428byte,
SHA47fc8a2d2413461782004c0b45e38cd0052f43e966b32d757ca486299e752d60.
Template concreti in `evidence/implementation-r001/product-inputs-r001/sequence-templates.json`;
launcher `after_freeze.py` nella medesima directory,10150byte,
SHAbe273524585ab9219382d866b54086cb11d2da542037f4c896a4dea175f31f95.
Request identifica argv/env/cwd e22ID in ordine. Non eseguire una stringa
ricevuta con eval; usare argv strutturati, shell=False/close_fds=True/env puro.

Il comando concreto per ogni ID è l'argv:

```text
[
  "/home/davide/workarea/markdown-for-llms/.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12",
  "-I", "-B",
  "/home/davide/workarea/markdown-for-llms/temp/run-a001-fase0-uv/evidence/implementation-r001/product-inputs-r001/after_freeze.py",
  "--operation", ID, "--attempt", "1"
]
```

CWD radice; environment pubblico chiuso da scope/prefix_operation, guardia
filesystem origine/config/site prima degli interpreti. Non mischiare os.environ.
Il launcher verifica snapshot/inventory, deriva target/startup reali, passa R/D4,
preflight e workload nella stessa invocazione run_offline/Firejail net=none,
stesso interprete e namespace dei figli. Dopo un tentativo, raccogli sempre
sessione/processi/receipt prima del seguito. Suffissi tentativo nuovi per retry.

1. `S-frozen-product`: produce S ufficiale dopo R/D4, riferimento a questo
   snapshot e input reali, `--uv` assoluto. Nessun S standalone come sostituto.
2. `B-native-sdist-and-wheel-from-sdist`: build uv nativa offline nel backend84
   esistente con no-build-isolation/constraints84/supply locale. Attesta startup
   prima dell'avvio, import backend e get_requires effettivi prima della build;
   trace raw dei veri build_sdist/build_wheel. Pre-import da solo o receipt post
   non provano gli hook nativi. Nuovi requisiti/backend ignoti: non acquisirli
   implicitamente. Conserva output univoci; uv default wheel dalla stessa sdist.
3. `B-archives-check`: audit tar/ZIP/metadata/RECORD/diecihash vs S corrente.
   Bind nomi/hash ai veri output B, non agli esempi di filename dei template.
   Nessuna installazione prima di audit PASS.
4. Per root, base-a e base-b: ID `native-base-sync-*`,
   `canonical-wheel-reinstall-*`, `I-ten-hashes-and-origins-*`,
   `E-current-preflight-*`, nell'ordine della request. Sync runtime locked
   offline no-install-project/no-build, poi stessa canonica no-deps/no-build;
   base senza setuptools/dev/API/ML. I verifica diecihash/RECORD/origini e B/S;
   E confronta corrente/S/snapshot/receipt, niente import applicativo o suite.
5. `locked-base-export`, `outside-clone-managed-venv`,
   `outside-clone-hash-locked-runtime`, `outside-clone-canonical-wheel`,
   `outside-clone-pip-check`, `outside-clone-I`, `outside-clone-E`:
   export hash-locked dal lock, venv esterna managed, solo runtime e canonica,
   pip-check e I/E da CWD esterno. Target `/tmp/run-a001-product-wheel-env-r001`.

I22ID e l'ordine effettivo sono in request/sequence; l'elenco sopra raggruppa
le dipendenze e non inventa output. Rootvenv/base-a/base-b/esterno/S/dist erano
assenti alla ricezione; controlla prima della prima creazione, non imporre
assenza agli output già creati dalla sequenza. Nuovi R/D4 reali per i workload;
non trasferire PASS dai runner precedenti o dallo scratch pubblico.

## Autonomia, identità e costi

R013/protocollo: mezzi tecnici reversibili non enumerati entro criteri/costi,
diagnosi e fix di reader/launcher/sintassi/deadline/log nella stessa chat.
Registra chiamata reale, errore, versione/delta e prove ripetute; parametri
adattabili non impongono nuovo mandato. Mantieni i bytes congelati: una nuova
versione di launcher operativo può stare in un path esclusivo con propria
identità; non attribuire il suo risultato ai bytes della versione congelata.
Se cambia un input della prova (diagnostico S/B/I, backend, modulo, lock,
config/fixture), correggi entro scope, raggruppa input completi e request della
label successiva per freeze ufficiale; non fermarti per chiedere il permesso
di correggere e non eseguire ufficiali con snapshot STALE. Nessun nuovo piano
per un errore ordinario. Non cambiare dieci moduli estranei a questo mandato.

Pool **112MiB cumulativi** da Hentry553541632, Delta<117440512 e
H+max(0,117440512-Delta)+16777216<939524096, H=max(logical,allocated) lstat
nofollow su run/.venv-python/.venv e /tmp esterno. Pool1GiB, stop896MiB;
libero repository>=1GiB e /tmp>=128MiB. Misura di consegna e residuo byte
in final-checks/response supervisore; rimisura prima e durante il seguito.
Vecchi costi inclusi e residuo32MiB non sommato; freeze non riapre112MiB.
Niente cleanup/reset/cache/tmp fuori ledger. Monitor0,5s/gap target1s,
non atomico; file32MiB/JSON8MiB/stream1MiB, log normali, paid0.

Workload7200s totale prefix+seguito: autore ha addebitato45,72075647953898s,
residuo7154,279243520461s prima delle prove, nessun workload aggiunto dal
supervisore. Sottrai ogni nuovo tentativo/guardia reale; singolo<=900s entro
residuo, outer+180. Le stime per ID non sono tetti non adattabili; non aumentare
il cumulativo. Conserva FAIL/parziali e raccogli processi propri prima dei retry.

Nessuna rete per queste prove. Mancanza supply offline: diagnosi, non download
implicito. Niente acquisizione Python/backend nuovo/dev/API/ML/pesi/font/paid,
nessun documento privato/import app/CLI/suite/V0comparativo/host policy/privilegi
o daemon operativo. Escalation per nuovo requisito/costo/fonte logica,
baseline alterata, privacy/confinamento non rispettabili, irreversibilità,
rifiuto sandbox o impossibilità dimostrata. Continua lavoro indipendente coperto.

## Consegna e limiti

Consegna report-r013 reale, evidence/delivery con22esiti/exit/input/output/costi
e S/B/I/E realmente prodotti; nessun PASS a file soltanto presenti. Se fallisce
un'operazione, correggi ordinariamente nello stesso mandato, senza inventare
PASS o proseguire su un prerequisito FAIL. Spiega eventuale blocco sostanziale.

Checkpoint autore incluso nel freeze: mantienilo invariato durante le prove.
Dopo verify finale preservane i bytes congelati e aggiorna il checkpoint per
la consegna; dichiara il solo delta documentale e identità, senza attribuire
MATCH dopo tale modifica. Gli esiti restano legati al freeze e ai loro input.
Stato comune/arbitrati/events/snapshot restano al supervisore.

Il PASS S/B/I/E base non chiude V1 config-only/flag/stale/ripristino né
V3–V6 import/fiveCLI/fasi/dev/API/suite, V0 nuovo/comparativi, V9/negativi.
Dev/API richiedono costi e ingressi distinti prima della loro preparazione;
collection non installa per ripararsi. V7/V8 pesanti obbligatorie a costo
separato; V10/V11 esclusi. Baseline62pass5fail/perdite, FAIL CRLF/racepyc,
s009byteFAIL/bootstraplacuna e s013logcap/receiptassente conservati.
ABI iniziale managedLinuxx86_64; lock universale non prova tutte le ABI.
Due review reali ChatGPT/Claude in nuove chat indipendenti sullo stesso
snapshot finale, arbitrato/GO finale futuri. Nessun commit/staging/merge/push/
promozione/deploy/cleanup o servizio automatico da attendere.
