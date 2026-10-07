# Implementazione r001 — policy lock offline ricevuta, prompt41

2026-10-06. **WAITING_FOR_SUPERVISOR_RECEPTION**. Implementatore esclusivo.
Una chiamata reale al wrapper, exit2 per **FAIL validazione argv nel helper
congelato**: scope passa --timeout900 ma run_offline.py ammette solo120s.
R/D4/uv/lock/check **NOT_EXECUTED**. Nessuna receipt nativa wrapper fabbricata.

## Identità e input

Request182507byte SHA13459712f318698b52c479ae66eafdaaa2cb42db6bd93b7c7d21de97e19e4fd4;
scope54162byte SHAcfc70c5ea9cd4f2b4dab5e39d038627fa16ee98b99e087f0101cb37ad7f8cd34;
snapshot122603byte SHA73ce0b5377765670fa5ff9fb8bc359c13615046d5924578f26e67d476368490f.
Worktree ecd3aaef7b6a8aee01175f5e045b34aa421f9e62ff34efbee99a975f6e5eae2a,
123file/360artefatti. S011 MATCH ingresso/pre-post/consegna; piano/arbitrato r003
SHA462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b /
f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d invariati;
r009 SHA02ef094dc4218778d8d1d6f477e8f66230aba64489dc19fa68d4c341f724e8c7.
S010 storico per cinque metadata,310artefatti intatti; non richiesto vecchio MATCH.
Report-r006/checkpoint/delivery preservati in before/; nessun registro condiviso.
Stessa copia16input s010, solo dependency-metadata ebooklib0.18/lxml/six:
byte e confronto semantico verificati; root/vecchia copia/pin/Marker/indici/
extras/conflicts/grafo invariati. Root uv.lock/.venv assenti.

## Esecuzione e blocco concreto

Request.proposed_operations e scope.operations identiche. Launcher ignorato
adattato in nuove evidenze a r009, syntax/formula controllate senza nuova
campagna sintetica. Bootstrap pyenv3.12.3 -I -B, env esterno chiuso pubblico;
wrapper invocato argv/cwd/environment_launcher esatti, shell=False/close_fds=True.
Native recipe AST è execve esatto, env R16 e uv19 coerenti;114 divieti package,
offline/keyring-disabled e no-downloads verificati senza eseguire il nativo.
Non chiamata allowlist uv generale; niente comando online.

Chiamata/sessione24972 raccolta, exit2, durata0.23164s, stop_reason null.
Stderr raw: `run_offline.py: error: Deadline figlio deve essere <=120s`.
Main del helper, righe291–292: `if not 0 < args.timeout <= 120:` precede
lettura target e outside(), quindi precede creazione output/R/Firejail/uv.
Nuova directory wrapper lock-offline non creata, receipt/inside/runner/loguv
non prodotti. I soli log/receipt esistenti sono del launcher d'autore.
Nessun rifiuto sandbox, timeout di risorse o FAIL del resolver inventato.
Scope impone900 e argv esatti, helper frozen rifiuta900: non correggibile
mutando il launcher senza violare quegli input. Non applicato patch al helper,
non sostituito900 con120 a runtime, nessun retry o auto-freeze.
Il secondo wrapper richiede lock exit0/audit: **NOT_EXECUTED** per dipendenza.
Audit TOML/hash/fonti/EbookLib/markers di un lock reale NOT_EXECUTED perché assente.
Questo esito non testa ancora l'efficacia della policy corretta r009.

## Controlli indipendenti completati

Gate cache prima/dopo: tutti114nomi simple-v20/**/*.rkyv coincidono esattamente
con policy.index_names, senza link/speciali nell'indice. Sdists-v9/builds-v0/git-v0
assenti o soli marker tecnici .git/.gitignore regolari ricevuti; zero corpi source,
Git/build riusabili. Snapshot identità di tutte le cache file, delta zero
hash/type/size/allocated; atime/mtime non attestati. Cache originale mai forgiata
o spostata. Nessuna lista ampliata né source locale/URL aggiunto.
Startup bootstrap/managed site-packages+stdlib enumerati prima del primo probe
e dopo la chiamata, zero .pth/sitecustomize/usercustomize; a1_coverage.pth baseline
esatto, variabili coverage/PYTHONPATH escluse. Origini bootstrap3.12.3 e managed
3.12.13/-I-B osservate, BUILD20260310/hash uv/managed/constraints conformi.
Config/credential scope solo assenza, contenuti non letti/stampati, env chiuso.
Target/lock baseline1892identità/211directory PASS via sola helper.verify.
Processo wrapper proprio raccolto, nessun figlio R/uv nato; nessun processo
altrui terminato. Zero rete/body/costi remoti/backend/build/install/import app.
Nessuna modifica host/rete/daemon/privilegi/sysctl/AppArmor/profili persistenti.

Ledger r009 run+.venv-python/Hentry525762560byte, lstat include dir/linkmetadata
senza follow/hardlink perpathname.16MiB nuovi(8attività8registri)+16esterni;
residuo contata una volta, max1GiB/stop896/libero1GiB. Monitor0.5s/gap target1s
non atomico; chiamata corta meno di un periodo, nessun picco istantaneo dichiarato.
Attività conservativamente Delta intero; registri verificati separati8MiB.
RLIMIT_FSIZE32MiB/file, stream1MiB/JSON8MiB; costi/freeze/Git/guardie finali in
final-checks.json. Nessuna cleanup/spostamento storage per passare i gate.

## Input concreti per il seguito, non autorizzati

[Richiesta — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md):
soluzione minima primaria è ricevere una nuova label/scope con timeout120
compatibile col helper intatto, output r002 distinti e stessi native argv/env/
policy/cache/input. Diff argv preciso conservato; deadline120/outer300
da ammettere esplicitamente, non usato per aggirare il comando corrente900.
Se900 è essenziale, patch di sole due righe del bound CLI proposta nel file
proposed-wrapper-deadline-r001.patch: cap massimo900, default120 invariato,
stessa gestione timeout nativo e outer child+180. AST della variante verificata
senza eseguirla; hash base/candidate e patch nel JSON. Richiede autorizzazione
del delta tracciato e nuova identità/freeze del helper; non applicata.
Il supervisore sceglie/riceve il fix ordinario operativo, senza riesame del piano;
stessa chat può proseguire. Questo è un risultato reale fermato da input
incompatibili, non una consegna volontaria di sola sintassi/preparazione.
Proposta NOT_AUTHORIZED/ready_for_execution=false. Nessuna acquisizione/build
implicita: se il vero resolver poi manca cache/altro source, resta un FAIL ricevibile.
Proposte registry backend s009/s010 respinte restano byte-identiche, non rilanciate.

## Consegna e limiti

[Completion — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[delivery — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[checkpoint — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md). Trasferire file reali temp/
managed/cache/work e working tree, non soltanto manifest. Git feature/run-a001-uv,
HEAD/dev/base66ba822, indice vuoto e nessun tracciato cambiato da implementatore.
Baseline originale completa caratterizzata/suiteFAIL62pass5fail/perdite e
S009 byteFAIL/prebootstrapmissing/rawhooknonobserved preservati, non ripetuti.
Lock rimane FAIL storico, nuovo uv NOT_EXECUTED; check NOT_EXECUTED;
product S/B/I/E/V0–V9 aperti, V7/V8 pesanti obbligatorie/costi distinti,
V10/V11/pesi/font/inferenza esclusi, local_marker non convalidato.
Due review indipendenti reali ChatGPT/Claude e arbitrato finale ancora necessari.
Nessun GO finale/product build/install/sync/test/Docker/Git/deploy/cleanup.
Prossimo supervisore prompt05+r009/checkpoint corrente; nessun servizio da attendere.
