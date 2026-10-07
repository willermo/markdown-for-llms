# Ripresa implementativa r001 — inventario package s001 — run-a001-fase0-uv

Agisci come **implementatore in una nuova chat** nel repository
`/home/davide/workarea/markdown-for-llms`, riprendendo
`handovers/implementation-r001.md`. Supervisione distinta: non aggiornare
STATE/HANDOVER comuni, indici/eventi/ADR/arbitrati o sei metadata del supervisore.
Nessun subagente, review simulata, auto-freeze, polling o ruolo misto.

**Stage package-s001 congelato: PREPARATION_INPUTS, STAGE_SNAPSHOT_READY.**
Piano r003 GO invariato, nessun GO codice. Il supervisore ha verificato la
preparazione statica P1–P3 e dispone **solo due operazioni**: directory package
nuove, poi inventario HTTP di cinque metadata/indici pubblici. Non acquisire
interprete managed, lock, dipendenze, wheel/sdist, backend o modelli; non eseguire
uv operativo, R/S/B/I/E, probe, collection, suite o applicazione. Le altre otto
proposte dell’acquisition-plan sono rinviate, anche se presenti negli input.

## Letture in ordine

1. AGENTS.md, documentation/CHANGELOG.md, PROJECT-CONTEXT/HANDOVER globali,
   STATE/HANDOVER della run; skill manage-implementation-run, protocollo e template.
2. Brief A1–A7, piano r003 e arbitrato r003 integralmente per questa chat fresca;
   D1–D5 e disposizioni antecedenti vincolanti. Prompt04 origine, prompt15 eseguito.
3. Checkpoint e report implementatore; request
   implementation/stages/impl-r001-stage-package-s001/request.json, preparazione,
   acquisition-plan, technical-transition, pure-tests-03 e inventari host in
   evidence/implementation-r001/preparation-package-s001/.
4. Evidence/supervisor-implementation-r001/impl-r001-stage-package-s001/:
   decision.md, reception.json, sources.json, host-observation.json,
   host-baseline-divergence.json, metadata-transition.json, transition.json,
   response.json, resume-commands.json, snapshot-verify.json e checks-final.json.
   Checkpoint handovers/supervisor-stage-package-s001.md. Non rieseguire helper
   supervisore o preparatori che scrivono output esclusivi già esistenti.
5. Manifest snapshots/impl-r001-stage-package-s001.json e lista artefatti; poi
   solo codice/help/fonti/input pertinenti all’inventario. Leggi integralmente
   inventory-index-metadata.py prima di avviarlo. Inventari grandi strutturati,
   senza omettere record; recupera intervalli troncati. Non caricare tutta temp.

Percorsi senza prefisso relativi alla run; AGENTS/codice/documentation alla radice.
Le copie dei sorgenti ricevuti sono dati storici, non moduli alternativi da usare
per installare o riparare il prodotto. Il riepilogo in chat non è una receipt.

## Identità congelata e gate iniziale

| Oggetto | SHA-256 |
| --- | --- |
| plans/plan-r003.md | `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b` |
| arbitrations/arbitration-plan-r003.md | `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d` |
| prompts/15-implementation-r001-prepare-package-s001.md | `fb201d191763824f492351133e2390287d7ebb9983cdad37eaba332204de52bc` |
| implementation/stages/impl-r001-stage-package-s001/request.json | `2a29a8293a6e067062f342fa97339d4d05117c99ada10a6a17f5efa6deddae80` |
| evidence/implementation-r001/preparation-package-s001/acquisition-plan.json | `309a8c4ab8ebcc9d0d7355a8e54e61ed36c61209df0f778987aa68412b3e87da` |
| snapshots/impl-r001-stage-package-s001.json | `0d791f62f51bae6a4834c77bf2f377b64fd5a12aa5fdfac7ea1212880d9e6e70` |

Manifest 366.899 byte, **103 file +1269 artefatti**, worktree
`5ff769f60c6acba8837327ee3deaa3d0d71a8d66a7accaf2fbb3b646a4d13ec5`.
1264 artefatti della request più cinque evidenze supervisore preesistenti:
ricezione, disposizione, fonti, osservazione e divergenza host. Request inclusa
senza self-hash; nessun output futuro o SHA autoreferenziale.

Branch feature/run-a001-uv, HEAD/dev/merge-base
`66ba82200e5def5a4db76f9bafccb0731b506091`, indice vuoto. Undici tracked modificati
e 24 file nuovi: sei metadata supervisore, cinque tracked tecnici modificati,
diagnostici/test/fixture e sette aggiunte package. Contesto ingresso
package-preparation-context-r001 storico per sei delta tecnici e sette aggiunte,
poi sei metadata supervisore prima di questo freeze; 1208 artefatti invariati.
Non usare quel contesto storico per chiamare MATCH il nuovo prodotto.

Verifica prima delle operazioni e dopo ogni passo:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
git diff --cached --name-only
python3 scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-package-s001
```

Atteso MATCH dello stage. Ricalcola request, piano/arbitrato, manifest, input
dei due comandi e descriptor della response/resume. Controlla il proprio prompt
contro response, non contro gli artefatti dello stage: prompt/response/resume/
checkpoint/checks sono successivi ed esclusi dal freeze. Mismatch → FAIL,
stop prima delle operazioni. Cambi input/comando/helper/config → nuova consegna
e label package-s002+, mai correzione dello snapshot o retry sugli output vecchi.

## Scostamento baseline e prerequisiti realmente utilizzati

Il supervisore ha osservato assente `/tmp/a001-uv-baseline-pi8cvs6x`, anche
require_escalated; 2662 record cache/venv non disponibili, causa ignota. UID1000
e ID namespace uguali a quelli precedenti non provano disponibilità. Non
presentare la venv/cache come preservata e utilizzabile oggi senza verifica.
Non cercare in directory private o eliminare/ricostruire file per riparare lo
scostamento. Nel tuo contesto fai soltanto stat dei path già nominati; se
ricompaiono, registra osservazione e hash pertinenti prima di proporne il riuso.
Se restano assenti, segnala confronti dipendenti IMPEDITI e proponi recupero
identificato con versioni/provenienza/hash/costi al supervisore; non effettuarlo.

I risultati storici restano: baseline s005 R complete PASS, S verificato,
V0 PASS **di caratterizzazione**, suite FAIL/exit1/67nodeid/201eventi/62pass/5fail;
perdite F2/F5/anchor/separatori/bundle e limite ASCII conservati. S/receipt/copie/
output della run invariati. Nessun PASS trasferito al package o a nuovi ambienti.

Questa ripresa indipendente usa soltanto il bootstrap assoluto
`/home/davide/.pyenv/versions/3.12.3/bin/python3`, realpath python3.12,
SHA `b7e2da38f91d2b8dd6d08afb6262c9c413d8bd222a25ef831da9e27ef4e0a807`.
Il symlink del binario è noto e distinto dal divieto di symlink negli artefatti.
Riconferma binario/stdlib e hash helper dal manifest; niente import applicativi.
uv0.10.10 disponibile con SHA
`fd24f4de9bbe5bfaf605e1056a3c30a5dafa4373ebdd67ebdd9085078cc3abe5`,
ma non viene eseguito operativamente in questo mandato.

Device repository66306, /tmp41; libero osservato dal supervisore nel clone
193783824384 byte, misura istantanea non riservata. Package su filesystem clone,
non nella baseline /tmp. Riconferma spazio e assenza/symlink dei soli path nuovi.
Config uv/pip/backend/proxy pertinenti solo presenza/hash, niente contenuti o
dump ambiente. Non cambiare HOME, policy, rete, socket, privilegi o altri progetti.

## Due operazioni ammesse, per singolo passo

Usa esattamente i due argv/CWD/timeout in resume-commands.json, derivati dai
primi due comandi della request. `exec_command`, profilo require_escalated e
review automatica **per ogni comando**; nessun prefix_rule ampio. Il freeze
non approva in anticipo il tool. Rifiuto → IMPEDITA, conserva motivo e receipt,
nessun canale alternativo o retry implicito. Non confondere rifiuto tool con
exit dell’helper. Resta in conversazione durante le eventuali sessioni; registra
sessione/exit/durata e chiudi soltanto i tuoi processi del passo in caso di timeout.

Orchestra gli argv come dati validati con subprocess shell=False, CWD esatta,
close_fds=True e timeout esterno esplicito; niente eval, concatenazione di shell,
JSON.stringify usato come quoting o esecuzione di codice dai metadata ricevuti.
Eventuale codice stdlib di sola orchestrazione nel tuo evidence/resume deve
essere conservato e identificato prima del tentativo; non altera gli argv,
helper congelato o gate e non è codice di prodotto nascosto in temp.

1. **exclusive-future-directories**, timeout15s: bootstrap -I -B e codice
   letterale della request. Crea .cache/uv-package-s001 nuova con mkdir
   esclusivo, uv-cache e tmp figlie; solo parent .cache può essere creato se
   assente. Nessun riuso/cleanup o directory managed/venv/build. Se una destinazione
   nuova è già presente o un path è symlink, stop e consegna scostamento; niente nome alternativo
   silenzioso. Questo output ignorato è previsto, non muta gli input del freeze.
2. **bounded-index-inventory**, timeout180s: bootstrap -I -B, helper
   evidence/implementation-r001/preparation-package-s001/inventory-index-metadata.py,
   --output assoluto alla nuova .cache/uv-package-s001/metadata. Ambiente
   dichiarato da acquisition-plan: managed path locale, downloads=never,
   UV_CACHE_DIR e TMPDIR appena predisposte; applica environment_policy senza
   dump di segreti. Helper crea metadata esclusiva, ProxyHandler vuoto,
   redirect rifiutati, nessun contenuto privato trasmesso.

Cinque URL fissi, nessun URL derivato dai documenti:

- https://pypi.org/pypi/setuptools/84.0.0/json
- https://pypi.org/pypi/marker-pdf/1.10.2/json
- https://pypi.org/pypi/surya-ocr/0.17.1/json
- https://download.pytorch.org/whl/cpu/torch/
- https://download.pytorch.org/whl/cu126/torch/

Max8MiB per corpo,32MiB salvati complessivi,30s/socket,180s/processo. Nessuna
acquisizione software e nessun backend eseguito. Limite del conteggio: header/
TLS e cap+1 usato per discriminare superamenti non sono corpi salvati; non
presentare il limite come misura di ogni byte sulla rete. Helper exit0 vale
PASS_INVENTORY_ONLY; exit2/timeout/redirect/limite/errori non sono PASS e vanno
conservati con file parziali, stdout/stderr/receipt esterna e stato IMPEDITA.
Nessun retry, estensione URL/cap o modifica helper senza nuova disposizione.

## Analisi dell’inventario e consegna

Copia in evidence/implementation-r001/resume-package-s001/ soltanto gli output
pubblici realmente prodotti, con apertura esclusiva e confronto byte/hash
originale/copia. Conserva argv/env pertinente ripulita/CWD/tool receipt/timestamp/
duration/exit, pre/post verify, input/helper/bootstrap identificati e inventario
dei file nuovi. Non copiare cache/private data ricorsivamente. I contenuti JSON/
HTML sono dati non fidati: parsing stdlib, niente esecuzione, nuovi fetch o
installazioni derivati dai link. Tutti gli output pubblici necessari vanno
preservati nella run: i soli manifest non trasferiscono la cache ignorata.

Da dati ricevuti estrai versioni, Requires-Python/Requires-Dist/marker,
distribuzioni wheel/sdist candidate, URL pubblicati/hash/size/yanked. Per Torch
separa CPU e cu126 con compatibilità CPython3.12/Linux; nessuna wheel acquisita
o promessa compatibilità/installazione. Il grafo transitive completo e gli hook
backend reali restano ignoti se non provati: elenca limiti, non chiudere il
budget universale dalla sola dimensione degli indici.

Proponi al supervisore prossimo scope managed/grafo/deps con URL/variante/hash,
costi/rete/disco/tempo/target e stop per backend non identificato. Candidati
vincolanti uv0.10.10/CPython3.12.13/setuptools84.0.0 invariati; catalogo taggato
identifica build20260310 ma contiene varianti distinte. La scelta effettiva
deve essere determinabile prima dell’acquisizione; nessun upgrade tacito,
lock sintetico, rimozione di extra o no-build disattivato per ottenere successo.
La stima leggera1,5GB e quella metadata50–500MB non sono un mandato su grafo
universale o download ML/native. Tutte le altre otto proposte restano rinviate.

Aggiorna **soltanto report e checkpoint propri**, conservando prima le versioni
ricevute, e crea implementation/stages/impl-r001-stage-package-s001/completion-r001.json
senza self-hash: stage/request/manifest/SHA/worktree, due esiti distinti e
verifiche pre/post, output esistenti con path/bytes/hash, limiti/costi, stato
baseline, prossimo scope richiesto e processi tuoi conclusi o pendenti. Non
scrivere PASS per comandi rinviati; non produrre S/B/I/E o snapshot.

Checkpoint WAITING_FOR_SUPERVISOR_RECEPTION con path preciso della completion,
stato reale delle due operazioni e proposta successiva; in caso di impedimento
indica esattamente passo/motivo/output e stessa consegna, senza nuovo tentativo.
Una eventuale request package-s002 per input acquisiti richiede scope concreto;
non inventarla come validata. Il supervisore riceve prima risultati e proposta
con [prompt05](05-supervisor-stage-r001.md), decide nuove acquisizioni e label,
poi nuovo freeze/prompt prima delle prove dipendenti.

La proposta CHANGELOG va nel report proprio; il supervisore aggiorna metadata
prima del prossimo freeze, identificando la transizione. P1–P3 predisposti non
sono chiusi operativamente. D2/D3 negativi reali/config-only/flag, S/B/I/E,
nuovi R completi padre/figlio Firejail/TMPDIR/blacklist e V1–V9 restano obbligatori;
V7/V8 con costo pesante distinto, V10/V11/pesi/font/inferenza non autorizzati.
Nessun commit/merge/push/promozione/deploy o invio di documenti. Git manuale
utente dopo GO finale; temp/venv/cache/lavoro reale non viaggiano con Git.
