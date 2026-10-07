# Implementazione r001 — completare cache metadata, lock e check

Agisci come implementatore nel repository `/home/davide/workarea/markdown-for-llms`.
Stessa chat ammessa. **AUTORIZZATO A IMPLEMENTARE nel perimetro ricevuto.**
Risultato completo: acquisizione diagnostica nativa dei metadata necessari,
cache verificata, lock universale originale offline sotto R/D4 e check.
Correggi errori ordinari e recupera ulteriori miss ammessi nella stessa chat;
nessun altro giro di sola preparazione o consegna per ogni nome mancante.

## Letture e identità

Leggi AGENTS, skill manage-implementation-run/protocollo aggiornato, STATE e
HANDOVER, checkpoint implementatore e report-r008. Recupera brief A1–A7,
indici architettura/decisioni/roadmap, piano r003 integrale e arbitrato D1–D5
se non già letti. Usa il contesto corrente, non tutti i prompt storici.
Percorsi seguenti relativi a `temp/run-a001-fase0-uv/`:

- `arbitrations/addendum-operational-protocol-r011.md`: disposizione corrente;
- `implementation/stages/impl-r001-stage-package-s013/request.json`;
- `evidence/supervisor-implementation-r001/impl-r001-stage-package-s013/`:
  authorized-scope.json, source-policy.json, metadata-prime.in, diagnostic-uv.toml,
  reception.json, transition.json, checks.json e freeze-verify.json;
- `handovers/supervisor-metadata-closure-r001.md`;
- `evidence/implementation-r001/resume-package-s012/`: delivery.md,
  actual-operations-r002.json, missing-native-metadata.json e next-gate-request-r001.json.

R011 riceve e corregge quella proposta: --no-deps salta i metadata wheel nel
resolver pinned, quindi il priming usa risoluzione transitive/binary-only.
R010 conserva la delega dei fix operativi. R011 supera nomi statici, divieto
assoluto di acquisizione e budget16MiB del mandato42 nei punti espliciti.
Nessun GO finale. PianoSHA `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`;
arbitratoSHA `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`.

S013 manifest133962byte/SHA
`8b97ef97bd732ad4966d7cd9ec7a2989d5ea6a8d369f5cc3d0254d0bcb176e1e`,
worktree `6c9ea856e35246b914afbe3b192e9a1c081b489d2f22018de90d922fd8da6a07`,
123file/404artefatti. RequestSHA
`8111c8833a0990872255e857b0af7a7c012a0218f36f2eda12757f12e8249c60`;
scopeSHA `b44e7f8f5622b7809dd419065a7430e9d1424dc1d1c7597cd68d51d2d7f6545b`.
Verifica hash pertinenti e, prima delle attività/alla consegna, da radice:
`python3 -B scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-package-s013`.
S012 ricevuto MATCH prima dei cinque delta documentali dichiarati; ora storico
per quei delta, artefatti intatti. Git feature/run-a001-uv, HEAD/dev/base
66ba82200e5def5a4db76f9bafccb0731b506091, indice vuoto. Nessuna modifica tracciata
in questo seguito. Preserva report/receipt/input s012 e copia checkpoint d’ingresso.

## Eseguire fino al risultato

Usa `scope.metadata_prime` e `scope.operations` come template strutturati:
argv/cwd/env pubblico espliciti, shell=False, close_fds=True, niente merge con
os.environ. Launcher Python assoluto pyenv3.12.3 -I -B. Work nuovo
`work/metadata-gap-native-r001` ed evidenze `evidence/implementation-r001/resume-package-s013`;
crea soltanto directory nuove previste, riusa cache/tmp/config e copia esistenti.
Non creare in anticipo gli output esclusivi del wrapper.

1. **Priming diagnostico online:** uv pip compile transitive, --no-build,
   --only-binary=:all:, --no-sources, config TOML vuota esplicita, managed3.12 noto,
   PyPI originale; 20 nomi iniziali, target metadata Darwin. Nessun --no-deps,
   --generate-hashes, install/sync/backend. Output requirements diagnostico
   escluso dal prodotto. Deadline fino a900s, esterna deadline+180.
2. **Audit e sigillo cache:** acquisizioni/URL/cache reali, startup/origine,
   niente source/build/Git salvo marker tecnici ammessi. Registra inventario/hash
   e lista esatta dei nomi con provenienza, poi divieti build per tutti i nomi
   cached salvo ebooklib, aggiungendo il progetto. Questo sealed-inputs è evidenza
   prima del workload, non un file futuro da includere retroattivamente nel freeze.
3. **Lock universale offline:** stessa copia originale16input, metadati osservati
   EbookLib0.18, pin/indici CPU/cu126/extra/grafo invariati. R/Firejail net=none/D4,
   uv nel medesimo namespace via execve. Aggiorna soltanto divieti derivati dalla
   cache sigillata; audita lock reale e nessun hook/build/install. Max900s.
4. **Check offline:** solo dopo lock exit0 e audit; stessa policy/R/D4, SHA lock
   invariato. Max120s. R PASS vale per la chiamata concreta, nessun trasferimento
   del PASS s012 al comando nuovo.

Una chiamata distinta per operazione; raccogli sessioni vive senza rilanciarle.
Monitora caps/tempi e processi propri; nessun kill per nome/processi altrui.
Priming/lock/check hanno massimi cumulativi di workload900/900/120s nel seguito;
validazione CLI fallita prima del workload non consuma quella deadline.

## Autonomia su metadata e strumenti

Source-policy.index_names identifica i114 nomi d’ingresso, **non** la lista
esatta richiesta dopo il priming. Sono ammessi i18 nomi nuovi già osservati e
la loro chiusura nativa, o altri miss PyPI del medesimo grafo originale, entro
64 aggiunte complessive. Confronta ogni cache nuova con trace/metadata pubblici
ricevuti. Nome/candidato ulteriore con provenienza e costi ammessi ⇒ acquisisci,
sigilla e riprova nella stessa chat; non richiedere nuova autorizzazione.

Adatta candidati/versioni/piattaforme del solo input diagnostico per i miss
osservati, salvando varianti locali e hash prima della chiamata. Non modificare
metadata-prime.in/config/scope congelati; le varianti restano evidenze identificate.
Non cambiare requisiti, Python, markers/environments, pin, fonti o extra del
prodotto, né sostituire una fonte dedicata con PyPI. I candidati diagnostici
scikit-learn1.9.0/narwhals2.26.0 non sono constraints del prodotto.

Il template132divieti è iniziale, non un vincolo di esattamente132nomi:
deriva i divieti da tutti i nomi effettivi sigillati. Mantieni Marker/setuptools/
progetto e la sola eccezione ebooklib0.18/lxml/six. Cache insufficiente dopo un
priming, anche parziale, si diagnostica e completa nei limiti ricevuti; audita
quanto è stato realmente acquisito, senza dichiarare PASS al priming fallito.
Nessuna cache forgiata/spostata o nuovo override dependency-metadata.

Correggi launcher/reader/argv operativi nella stessa chat. Mantieni la compattazione
execve riuscita in s012; valida che la forma compatta espanda negli argv/env reali
registrati e rispetti MAX_ARG_LEN4128. Output con suffissi esclusivi, errori e
parziali preservati. Non riscrivere prove precedenti, riprovare alla cieca o
azzerare budget. Un timeout reale richiede processi raccolti, gate validi e
ripetibilità prima del retry. Nessun freeze per ogni nome o parametro ammesso.

## Risorse, rete e arresti sostanziali

La scope distingue acquisizione HTTPS da lock offline: il priming non è R
net=none PASS. Host pubblici PyPI/files.pythonhosted.org, TLS, keyring disabled,
config/env chiusi, niente auth/proxy/.env/documenti. Verifica startup/origine
bootstrap/managed/baseline prima dei probe e dopo; baseline immutabile con
check_preserved.verify, nessun import app/native. Nessuna modifica rete/daemon/
socket host, privilegi/sysctl/AppArmor/profili persistenti o altri progetti.

Preferisci PEP658/range. Uv può ripiegare sulla wheel per leggere metadata:
wheel della chiusura diagnostica ammesse entro i caps, contandole realmente,
mai installazione/ABI. Niente sdist/Git/backend, pesi/modelli/font o archivi
binari completi torch/nvidia/Marker/Surya/OpenCV. URL/redirect fuori host o
payload fuori perimetro ⇒ ferma acquisizione, conserva ciò che è avvenuto.
Non dichiarare whitelist atomica o solo metadata-body se non dimostrato.

Tranche cumulativa **32MiB totali**, che sostituisce16, non aggiunge32:
Hentry525762560byte invariato, costi precedenti inclusi;24MiB attività/8MiB
registri,16MiB esterni,pool1GiB/stop896MiB/libero1GiB. Formula/caps in scope;
ledger run+.venv-python, lstat senza follow, file32MiB/log1MiB/JSON8MiB,
monitor0,5s/gap target1s non quota atomica. Stima body4MiB non cap né misura
wire; zero servizi a pagamento. Niente cleanup/reset/storage fuori ledger.

Arresti sostanziali: risorse/tempo/nomi oltre limite, fonte o requisito nuovo,
conflitto del grafo dimostrato con dati sufficienti, source/backend necessario,
isolamento/privacy/integrità non rispettabili, input protetti difformi o
rifiuto automatico sandbox. Blocca la sola attività dipendente, preserva
ragione/azione e continua lavoro indipendente ammesso. Gli errori CLI/reader
ordinari e ulteriori miss entro l’ammissione non sono motivo di consegna.

## Consegna

Scrivi `implementation/report-r009.md`, completion s013 legata a request/scope/
manifest e `evidence/implementation-r001/resume-package-s013/delivery.md`:
actual argv/cwd/env/sessioni/tempi/exit/log/receipt, cache/provenienza/sealed names,
startup/integrità/budget, R/wrapper/priming/lock/check separati e audit/hash lock.
Aggiorna solo report/checkpoint implementatore, con copia d’ingresso; nessun
registro condiviso o snapshot autore. Se lock/check riescono consegna insieme
richiesta concreta per promozione/S/B/I/E, senza eseguirli implicitamente.

Baseline caratterizzata/suite62pass5fail e perdite, s009 byteFAIL/lacune restano.
S/B/I/E e verifiche prodotto aperti, V7/V8 pesanti distinti e V10/V11 esclusi.
Due review reali ChatGPT/Claude e arbitrato finale necessari. Nessun GO finale/
commit/merge/push/deploy/cleanup; Git manuale. Stato WAITING_FOR_SUPERVISOR_RECEPTION,
prossimo supervisore prompt05+r011 e checkpoint corrente. Nessun servizio da
attendere automaticamente; trasferire file reali ignorati e lavoro non committato.
