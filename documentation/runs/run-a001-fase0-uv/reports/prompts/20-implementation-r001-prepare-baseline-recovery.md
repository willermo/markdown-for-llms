# Implementazione r001 — preparare il recupero baseline dopo metadata s002

Agisci come **implementatore in nuova chat distinta**, repository
`/home/davide/workarea/markdown-for-llms`, run `run-a001-fase0-uv`.
Mandato: **sola preparazione del recupero baseline**, senza acquisizioni o
installazioni. Non sei supervisore/revisore; nessuna delega o roadmap automatica.
Piano r003 GO/D1–D5/A1–A7 invariati, NO_GO r001/r002 conservati, nessun GO codice.
Non chiedere nuovamente approvazione uv; prepara una request concreta D1.

## Letture e identità

Leggi AGENTS.md, skill manage-implementation-run, protocollo run-lifecycle,
CHANGELOG/indici/roadmap; temp/HANDOVER.md/PROJECT-CONTEXT.md e STATE/HANDOVER della
run. Poi checkpoint autore corrente handovers/implementation-r001.md, brief,
piano r003 integrale, arbitrato r003 D1–D5/matrici antecedenti e prompt04.
Prompt17/17a/19 sono storia conservata, non autorizzazioni a ripeterne le operazioni.
Recupera porzioni troncate, non caricare tutta temp/.

Percorsi seguenti relativi alla run:

- `implementation/stages/impl-r001-stage-package-s002/completion-r001.json`,
  SHA256 **fac4b8d9196f8aad2ec721b9bc5a80f44a1f64ce57e46f6f21227af9b2db3af8**,
  131343byte, PASS_METADATA_ONLY ricevuto dal supervisore.
- `evidence/implementation-r001/resume-package-s002/`: delivery.md,
  metadata-summary-r001.json, baseline-recovery-proposal-r001.json,
  managed-next-gates-r001.json, receipt e copie dei raw.
- `evidence/implementation-r001/resume-package-s002-preflight-r001/`:
  audit/tool/delivery-closure, primo errore lettore e r002, ingressi preservati.
- `evidence/supervisor-implementation-r001/package-metadata-reception-r001/`:
  **decision.md e next-scope.json sono la disposizione corrente**; reception.json,
  metadata-audit.json, author-reader-diff.txt, transition.json e response.json.
  Ricevuta della completion e preparazione nuova sono passaggi distinti.
- `evidence/supervisor-handover-package-s002-r001/storage-decision.md`,
  storage-scope-r001.json e prompt17a per destinazione temp/work.
- Per originali baseline: evidence/implementation-r001/baseline-originals/,
  baseline-inputs.json, baseline-build-inputs.json, baseline-requirements.txt,
  baseline-environment.json, baseline-freeze.json/txt e request/sources/output
  s005. Leggi strutturalmente i grandi inventari, senza chiamare presenti i2662
  record esterni indisponibili. Non usare i moduli nuovi P1–P3 come originali.

Piano SHA256 `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`;
arbitrato SHA256 `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`.
Stage s002 storico SHA256 `df7d3f60255dded579fa315cbc05dfb98d24f5f0a301ab2f884d3a9cbeeee809`,
worktree `0f682ad078e1011805af78c410ebe89edc06416682406e55610ffc814fc86e84`,103/1516.
È stato MATCH durante i tre passi e alla ricezione; dopo ricezione cambia soltanto
per sei metadata supervisore.1516artefatti e manifest originario restano invariati.

**Nuovo contesto: baseline-recovery-preparation-context-r001**,
`snapshots/baseline-recovery-preparation-context-r001.json`. SHA/worktree/count
esatti in response.json e handovers/supervisor-package-metadata-reception-r001.md,
scritti dopo il freeze; questo prompt ne è input, nessun autoSHA/ciclo.
Prima di modificare: git status --short --branch, branch --show-current,
rev-parse HEAD dev, merge-base HEAD dev, diff --cached --name-only, diff --check;
poi verify del contesto nuovo con scripts/run_context.py. Ricalcola SHA del
manifest e verifica la corrispondenza con response, non solo MATCH.
Attesi feature/run-a001-uv,HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto,11tracked modificati/24file nuovi. Nessun delta tecnico autorizzato
in questa preparazione. Eventuali differenze ulteriori STOP al supervisore.

## Stato ricevuto e limiti

Tre passi s002 exit0: directories PASS_DIRECTORIES_ONLY;5sidecar128050byte e
17JSON733054byte PASS_METADATA_ONLY,22HTTP200/**861104byte**.90copie byte/SHA
identiche,220identità/97input/1498hash/5host/19config riconfermati dal supervisore.
LANG C.UTF-8 contro en_US.UTF-8 preparato è ereditarietà prevista nel driver;
delta esplicito accolto solo per metadata, non equivalenza universale per V0.
Primo FAIL lettore autore KeyError operations e lettore supervisore sul nome
directory conservati con versioni corrette; nessun retry HTTP o helper mutato.

122Requires-Dist,513distribuzioni/25candidati nominali; metadata non sono archivi
verificati. Marker/Surya differiscono testualmente da info; nessun resolver
dimostra equivalenza semantica. CPU/cu126 separati, size Torch ignota.
Regex2026.9.29 della baseline non soddisfa il vincolo Marker regex<2025:
mantieni baseline e produzione distinte, non cambiare pin per fare risolvere.

Baseline `/tmp/a001-uv-baseline-pi8cvs6x` e venv/tmp/workspace/uv-cache/tiktoken-cache
assenti;2662record non disponibili. Sola stat di questi path se necessaria,
nessuna scansione privata di /tmp/altro progetto. Causa probabile dichiarata
pulizia /tmp, non dimostrata. Confronti **IMPEDITI**. V0s005 R/S/PASS
caratterizzazione e suiteFAIL/exit1/67nodeid/201eventi/62pass5fail, perdite
F2/F5/anchor/separatori/bundle/limiteASCII conservati. Nessuna venv ricreata
equivalente per intero o PASS trasferito. Nessuna correzione di algoritmi/test legacy.

## Output della sola preparazione

Crea evidenze proprie nuove sotto
`evidence/implementation-r001/preparation-baseline-recovery-r001/` e request
`implementation/stages/impl-r001-stage-package-s003/request.json`.
Se già esistono, preserva e identifica prima gli scostamenti: non sovrascrivere.
Copie versionate del report/checkpoint autore ricevuti prima di aggiornarli.

Target **futuro**, ancora assente:
`temp/run-a001-fase0-uv/work/baseline-recovery-r001/`, con wheels/tmp/uv-cache/
tiktoken-cache/venv/workspace e eventuali output ordinati del driver.
Verifica realpath/device/spazio/ignore/antenati e assenza iniziale, **non crearlo**.
Parent work è ora presente per s002: non richiederne l'assenza storica e non
modificarne gli output. Non creare prepared-inputs nel target per comodità:
helper/manifest/requirements stabili stanno nelle evidenze proprie congelabili;
eventuali copie future vanno identificate come output del passo directory.
.venv/.venv-python del prodotto e output s001/s002 restano inalterati.

Realizza un pacchetto di preparazione reviewabile, con:

1. **Allowlist esatta** derivata da next-scope.json:17wheel proposte, relativi
   URL/byte/SHA/pin/Requires-Python/tag/provenienza raw e18°URL tokenizer.
   Tutti17SHA nelle513impronte storiche, size totale wheel4874528byte;
   tokenizer1681126byte SHA256
   `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`,
   URL `https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken`,
   cache key `9b5ad71b2ce5302211f9c61530b329a4922fc6a4`.
   Nessun latest/ricompilazione requirements/resolver per sostituire i17pin.
2. **Acquisitore stdlib separato e driver**, solo18GET su files.pythonhosted.org
   e openaipublic.blob.core.windows.net, URL hardcoded dal manifest verificato.
   Size/hash esatti, cap per corpo pari alla size pubblicata, massimo un byte
   discriminante;6555654byte corpi complessivi, non traffico wire.30s/socket,
   600s subprocess massimo proposto. No redirect/proxy/credenziali/retry/HEAD/
   Range/fallback/sdist. Non riusare i cap1MiB metadata: due wheel e tokenizer
   lo superano. Nuovi helper non modificano quelli s002 congelati.
3. **Verifica locale degli archivi prima di esecuzione/installazione**:
   ZIP senza estrazione cieca, path assoluti/../duplicati/symlink/file speciali,
   limiti numero voci/byte decompressi e cumulativi, CRC/digest/dimensioni;
   Name/Version/Requires-Python, WHEEL/tag/Root-Is-Purelib e RECORD coerenti,
   eccezioni RECORD nominate. Registra .data, script e file .pth/startup presenti:
   non eseguirli durante l'analisi, non dichiararli innocui per estensione.
   Nessun hook/backend o import dei moduli acquisiti. Sdist/backend ignoto STOP.
   Dimensioni compresse non sono un limite all'espansione: rendi concreto il
   tetto prima di consentire venv/install. Non estrarre nulla in questa chat.
4. **Requirements selezionati hash-required** per sole17wheel, con nomi/file
   locali predeterminati e mapping agli URL, senza nuove dipendenze dalla rete.
   Chiusura dipendenze da verificare sui METADATA delle wheel dopo acquisizione,
   rispetto al profilo baseline e marker/extra/3.12.3; non inferirla dal solo
   totale17 o da --no-deps. Progetta confronto degli hash/metadata e stop sui
   requisiti attivi insoddisfatti prima dell'uso, più pip check futuro.
5. **Proposta venv/install offline**, distinta dall'acquisizione, con argomenti
   esatti e driver visibile, quote/tool review separate. Bootstrap assoluto
   `/home/davide/.pyenv/versions/3.12.3/bin/python3`, alias verso python3.12,
   SHA256 `b7e2da38f91d2b8dd6d08afb6262c9c413d8bd222a25ef831da9e27ef4e0a807`;
   venv via -I -B -m venv, pip24bundled2110226byte SHA256
   `ba0d021a166865d2265246961bec0152ff124de910c5cc39f1156ce3fa7c69dc`.
   Nessun upgrade/install nel pyenv originale. Disegna pip24 isolato, no-index,
   sole wheel locali hash-required/only-binary/no-deps/no-cache-dir, nessun
   resolver remoto; controlla semantica esatta su codice/help/fonti pinned.
   Pip bundled della venv non deve essere scaricato né sostituito tacitamente.
   Argv della proposta s002 erano interfacce future, non comandi già approvati.
6. **Origini e inventario dopo recupero**, ancora futuri: interpreter/stdlib/
   venv reale/pip/deps/RECORD/cache e17versioni, sorgenti baseline-originals
   riconciliati con manifest s005, fixture e argomenti storici. Registra
   Requires-Python e compatibilità tag rispetto all'interprete/host effettivi,
   non solo Linux nel filename. In questa preparazione usa inventari/fonti
   statiche già identificati; verifiche effettive sull'ambiente nuovo saranno
   passi futuri espliciti, non probe/applicazione anticipata.

Helper stdlib, argv strutturato/shell=False/close_fds, path root-relative nei
manifest e assoluti nei comandi. Aperture/directory esclusive, nessun symlink
negli antenati/output, lock contro due processi e marker persistente contro
retry dello stesso passo. Fai fallire prima di scritture/HTTP su input mutati.
Raw/partial e receipt sempre conservati; timeout termina/raccoglie soltanto
figli propri, log limitati senza byte remoti illimitati. Se receipt/copia non
riesce, registra incompletezza, mai PASS. Output assegnati dal driver, non da
nomi/path/comandi emersi nei metadata. No eval o import di METADATA.

Budget **di progetto da concretizzare, non autorizzazione esecutiva**:
500MiB disco aggregato, stima256MiB da motivare, include wheel/tokenizer/copie/
ZIP espansi/venv/cache/log/report e preparazione. Riserva receipt e budget per
file esterni al contatore; identifica file logici e allocati. Timeout120s venv,
180s install,600s acquisizione proposti: applicarli realmente nei driver,
distinguendo controlli/copie e tempo tool. Backend/tetto ignoto = STOP, niente
espansione del cap implicita. Nessun costo ML/pesi/cloud pagato incluso.

Environment proposto figlio dichiarato: HOME invariata, PATH bootstrap:/usr/bin:/bin,
TMPDIR/cache/tokenizer nel target nuovo, UV_PYTHON_DOWNLOADS=never, nessun proxy/
credenziale/PYTHONPATH/PYTHONHOME/PYENV/PIP/SETUPTOOLS/UV override ereditato salvo
quelli esplicitamente fissati. Definisci locale deterministica per nuovi comandi;
non attribuire al padre tool l'ambiente che il driver imposta solo sul figlio.
Config utente/sistema/progetto presence/hash senza segreti, semantica reale
pip isolated e precedenza config da identificare senza esecuzione software.

Verifiche consentite ora: letture/AST/JSON/TOML/hash/Git, fonti primarie datate
su dubbi tecnici, piccoli test stdlib con stream HTTP e ZIP sintetici locali,
zero rete/socket/prodotto. Testano rischi reali: cap/size/SHA/redirect/parziali,
duplicati/evasioni/RECORD/espansione ZIP e stop; non specchiare l'implementazione.
Nessun vero download/wheel installata/ensurepip/venv/uv/pip operativo/Pandoc/
pytest/collection/app/tokenizer/runner o benchmark in questa preparazione.
Fonti già in run da leggere prima di aggiungere acquisizioni documentali; nessun
fetch degli18asset per verificare disponibilità. Preserva ogni FAIL e versione.

## Request e catena successiva

Prepara D1 schema1, run/revisione/stage package/label
**impl-r001-stage-package-s003**, subphase BASELINE_RECOVERY_INPUTS,
status **WAITING_FOR_STAGE_SNAPSHOT**, Git/branch/base, stabili+assenze,
host/config/originali/copie distinti, helper/manifest/argv/costi/stop/prove future.
Includi piano/arbitrato/prompt04/prompt20, scope/decisione/response e contesto
correnti, stage s002/request/completion/output pertinenti e loro provenienza.
I1516antecedenti devono restare identificati/invariati; includi nella lista
le evidenze ereditate necessarie, raw effettivi e copie già conservate senza
duplicarle ancora. Lista esatta di file esistenti confinati alla run, no
symlink/.. o path assoluti, request stessa senza self-hash. Snapshot s003,
venv/nuovi asset/S/B/I/E/receipt futuri non sono input esistenti da congelare.
Report/checkpoint propri correnti sono mutabili: usa copie versionate d'ingresso.

Questa è la preparazione di un **possibile** perimetro software di recupero,
non la sua esecuzione. Il supervisore riceverà la request e deciderà argv/gate
effettivi prima del freeze e del nuovo prompt operativo. Non autorizzarti dai
soli URL presenti. Non creare snapshot o auto-delegarti la supervisione.

Dopo l'eventuale acquisizione/ricostruzione, receipt e input nuovi torneranno al
supervisore prima del futuro **baseline-s006**: nuova R completa parent/child,
TMPDIR/daemon/blacklist/socketpair e nuovo S prima dei confronti P0/V0.
Un cambio path non prova equivalenza del confinamento; nessun R o PASS vecchio
trasferito. Non eseguire R/suite/probe/fedeltà da un eventuale comando install.
Sorgenti originali e risultati s005 restano separati dal nuovo prodotto.

Tutte8operazioni software di produzione precedenti restano RINVIATE: managed,
lock/lock-check/deps/backend/build/sync/install canonico. uv0.10.10/Python3.12.13/
setuptools84 invariati; selector GNU/build20260310/redirect/cap managed e grafo
universale ancora aperti. No-build/extra/CPU/cu126 invariati, no restrizioni
di piattaforma implicite per risolvere il lock. D2/D3/current↔S↔snapshot/
config-only/flag/freschezzaIDE e V1–V9, due review codice/arbitrato finale futuri.
V7/V8 obbligatorie/costo distinto,V10/V11/pesi/font/inferenza esclusi.

## Consegna e responsabilità

Report della preparazione nelle evidenze proprie, proposta CHANGELOG nel report,
request reale e aggiornamento di **soli** implementation/report-r001.md e
handovers/implementation-r001.md con WAITING_FOR_STAGE_SNAPSHOT e processi propri.
Non toccare sei metadata tracciati, prodotto/test/fixture/diagnostici esistenti,
STATE/HANDOVER comuni, indici/eventi/arbitrati/snapshot. Nessun nuovo GO.
Verifica link/JSON/stati/diff/hash e contesto prima/dopo, conserva gli errori.

Consegna al supervisore tramite prompt05, senza polling/servizio continuo.
Nessun cleanup/host/sysctl/AppArmor/socket/rete/setuid/profili/altro progetto,
commit/merge/push/promozione/deploy/invio documenti. Git manuale utente dopo GO
finale. Temp e dati ignorati non viaggiano con Git: conservare file effettivi.
