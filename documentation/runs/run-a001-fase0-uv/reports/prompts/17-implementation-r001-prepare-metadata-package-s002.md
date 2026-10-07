# Implementazione r001 — preparazione metadata package s002

Agisci come **nuovo implementatore**, chat distinta, nel repository
`/home/davide/workarea/markdown-for-llms`, run `run-a001-fase0-uv`.
Mandato **sola preparazione statica** del prossimo ingresso metadata s002.
Non sei supervisore/revisore; nessuna delega. Piano r003 GO con D1–D5 immutati,
nessun GO codice, NO_GO r001/r002 storici conservati. Adozione uv già autorizzata.
Non ripetere bootstrap/branch o chiedere di nuovo la sua approvazione.

## Recupero e identità

Leggi in ordine AGENTS.md, manage-implementation-run/SKILL.md, protocollo,
CHANGELOG e temp/HANDOVER.md/PROJECT-CONTEXT.md, STATE/HANDOVER della run,
brief A1–A7, **piano r003 integrale**, arbitrato r003 D1–D5 e prompt04.
Poi checkpoint autore handovers/implementation-r001.md, completion s001,
delivery e due proposte resume-package-s001; disposizione e next-scope propri
del supervisore package-inventory-reception-r001. Prompt16 è storico eseguito,
non un’autorizzazione a ripetere i suoi passi. Non caricare tutta temp/.

Percorsi tabella relativi alla run; ricalcola SHA256, recupera intervalli troncati.

| Artefatto | SHA-256 |
| --- | --- |
| plans/plan-r003.md | `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b` |
| arbitrations/arbitration-plan-r003.md | `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d` |
| prompts/04-implementation-r001.md | `fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035` |
| prompts/16-implementation-r001-inventory-package-s001.md | `0d7d3be6f48688c4664040eab398695c147cdd511cf15dd0cbafa132fc3c6c30` |
| implementation/stages/impl-r001-stage-package-s001/completion-r001.json | `c43ba8b66ea35b2e6742472293d01a0e05e05a715b9318796861b39a65f25ff6` |
| evidence/implementation-r001/resume-package-s001/next-scope-proposal.json | `c03d5667e3d1069c4b6fac843421a19a1bb511715cdcc6d380f50cac9b56a480` |
| evidence/implementation-r001/resume-package-s001/baseline-recovery-proposal.json | `e263e418d9f256fc30770af1c32b8469b7ba3d19fac74364f61443457ca6e427` |
| evidence/supervisor-implementation-r001/package-inventory-reception-r001/next-scope.json | `e6e5172b816e11c361680341def111b464ddbf7578ce411970ca645fcdc12275` |
| evidence/supervisor-implementation-r001/package-inventory-reception-r001/decision.md | `bf70d3032f26e35629d2d4496db3902c0234498fd3c3e5fe9e01e1e1178b769b` |

Contesto d’ingresso **package-metadata-preparation-context-r001**, prodotto dal
supervisore dopo i sei metadata: snapshots/package-metadata-preparation-context-r001.json.
Questo prompt ne è input; SHA/worktree/count esatti nella response e nel checkpoint
supervisore successivi al freeze, non incorporati qui per evitare autoreferenzialità.
Leggi `evidence/supervisor-implementation-r001/package-inventory-reception-r001/response.json`
e handovers/supervisor-package-inventory-reception-r001.md, confronta manifest e
ricalcola `python3 scripts/run_context.py verify run-a001-fase0-uv --label package-metadata-preparation-context-r001`.
Atteso MATCH prima preparazione; non sovrascrivere snapshot. Modifiche nuove autorizzate
in evidenze proprie rendono il contesto storico soltanto se sono suoi input;
registrare delta/newlabel, non mascherare divergenze tecniche o nuovi file Git.

Git atteso branch feature/run-a001-uv, HEAD/dev/merge-base
`66ba82200e5def5a4db76f9bafccb0731b506091`, indice vuoto,11tracked modificati/
24file nuovi (sei metadata supervisore, P1–P3 statici e diagnostici/fixture).
Verifica git status --short --branch, branch, rev-parse HEAD dev, merge-base,
diff --cached --name-only e diff --check. Nessun lock/managed/.venv presente.
Nessuna modifica al prodotto/pyproject/setup/pin/constraints/README/Docker/test/fixture/
diagnostici esistenti in questo mandato; i sei metadata comuni appartengono al supervisore.

## Stato consegnato

Package s001 **PASS_INVENTORY_ONLY**: due passi reali exit0,5HTTP200/640001byte
salvati e preservati,103file/1269artefatti MATCH durante le prove e alla ricezione.
Manifest s001 SHA `0d791f62f51bae6a4834c77bf2f377b64fd5a12aa5fdfac7ea1212880d9e6e70`,
worktree `5ff769f60c6acba8837327ee3deaa3d0d71a8d66a7accaf2fbb3b646a4d13ec5`.
Ora storico solo per sei metadata files/worktree_sha256,1269artefatti invariati.
La directory .cache/uv-package-s001 è output autorizzato già creato: non ricontrollarne
l’assenza iniziale, non riscriverne i corpi/copie. Errore lettore FAIL originale
preservato e versione r002 riconciliata, nessun retry operativo o helper/argv mutato.

89Requires-Dist/6distribuzioni PyPI,1219link CPU+354cu126: metadata pubblicati,
non software verificato. Marker1.10.2/Surya0.17.1 e Torch2.7.1 cp312/cp312/manylinux_2_28_x86_64
restano candidati, CPU/cu126 distinti. Grafo universale/transitive/tagcompat reale/
Torchsize/backend hooks ignoti. PyPI info non garantisce uguaglianza alla specifica
wheel METADATA; sidecar esatto serve al confronto. Nessun runtime è passato da s001.

Baseline /tmp/a001-uv-baseline-pi8cvs6x assente: root/venv/tmp/workspace/uv-cache/
tiktoken-cache e2662record non disponibili. Stat soltanto dei path nominati;
nessuna scansione privata, ricerca globale, cleanup o ricreazione in questo mandato.
Causa ignota. **Confronti dipendenti IMPEDITI**, non omessi o PASS ereditato.
V0s005 PASS caratterizzazione storico; suiteFAIL/exit1/67nodeid/201eventi/62pass5fail,
perdite F2/F5/anchor/separatori/bundle incompleto e limiteASCII conservati.
Nuovo target baseline futuro `.cache/uv-baseline-recovery-r001`, non crearlo ora.

## Preparare request s002 senza rete o acquisizioni

Directory proprie nuove `evidence/implementation-r001/preparation-package-s002/`
e `implementation/stages/impl-r001-stage-package-s002/`. Nessun output operativo
cache/venv acquisito ora. Target futuro metadata `.cache/uv-package-s002` nel
filesystem del clone, distinto dal s001; verifica assenza/symlink/realpath/parent,
git check-ignore e spazio reale. La regola .cache esistente basta: non modificare
.gitignore per questo mandato. Se target già esiste, fermarsi e consegnare scostamento.

Concretizza le **due sole future operazioni metadata** in next-scope.json del supervisore:

1. Cinque sidecar fissi (setuptools84,marker1.10.2,surya0.17.1,torch2.7.1+cpu/+cu126).
   Copia integralmente5URL e SHA256 attesi, verifica derivazione dai raw s001/attributi
   core-metadata o alias legacy e .metadata aggiunto a URL wheel senza fragment.
   1MiB/corpo,5MiB corpi salvati,30s/socket,180s/processo; hosts files.pythonhosted.org
   e download-r2.pytorch.org soltanto. Nessuna richiesta alle wheel/sdist come fallback.
2. Diciassette GET JSON PyPI a **versione fissa** della proposta baseline,17URL esatti
   dal perimetro. Verifica17pin/513hash contro baseline-requirements.txt storico;
   non cambiare versioni o risolvere latest.1MiB/corpo,17MiB salvati,30s/socket,
   600s/processo, host pypi.org soltanto. L’hash di questi nuovi corpi è sconosciuto:
   misurarlo/preservarlo dopo GET, non confrontarlo agli hash wheel dell’allowlist.

Disco metadata: stop96MiB totali tra raw/copie preservate/log/inventari/analisi;
stima motivata prima della request, corpi massimi22MiB complessivi. Non è un budget
per software/ML/venv. Cap sui body, escluse intestazioni/TLS e al massimo il byte
discriminante cap+1 per risposta; non chiamarlo tetto del traffico wire.
I due batch hanno receipt/costi/esiti separati; non dedurre PASS di uno dall’altro.

Realizza helper e driver **stdlib** visibili nelle nuove evidenze, con manifest
allowlist immutabile (URL/schema/SHA sidecar e pin/versioni baseline). Nessuna
lettura/esecuzione di URL/comandi/path emersi nei metadata. Path output assegnati
dal driver (ordinali), shell=False/argv strutturati, no eval. Output esclusivi,
rifiuto symlink/evasioni/preesistenti; lock contro doppia esecuzione dello stesso step.
HTTP GET soltanto su22URL congelati, proxy disabilitati esplicitamente, redirect
rifiutati, nessuna credenziale/header/cookie privato, TLS normale; una richiesta
per URL, nessun retry/canale alternativo. Errore/cap/hash sidecar/versione/schema/
yanked candidato/redirect = FAIL preservato e STOP, senza fallback o riparazioni.
Conservare partial ricevuti/limiti/finale anche se fallisce; timeout esterno termina
solo subprocess propri, niente processi di altri. Niente byte non limitati in log.

Analizzatore locale separato senza HTTP: raw immutati con SHA. Per sidecar usa
parsing email stdlib, conserva Requires-Dist ripetuti/marker/extra e Name/Version/
Requires-Python; confronto info JSON distinto, non equality obbligatoria arbitraria.
Per17JSON controlla nome normalizzato/versione esatta, enumera urls con filename,
size/hash/yanked/Requires-Python/tag; candidati wheel CPython3.12/Linux x86_64 o
py3-none-any con SHA presente nell’allowlist storica. Richiede almeno un candidato
ammissibile/non yanked per pin; i files di altre piattaforme non sono un FAIL
automatico. Non scaricare né installare i candidati. Vincoli Requires-Python/tag
restano metadata fino a valutazione completa, non compatibilità runtime dichiarata.
File METADATA/hook/import mai eseguiti. Sidecar non fornisce size della wheel:
non dedurla dalla size del metadata o fare HEAD/Range non inclusi nella request.

Scegli bootstrap assoluto già identificato `/home/davide/.pyenv/versions/3.12.3/bin/python3`
con -I -B, SHA `b7e2da38f91d2b8dd6d08afb6262c9c413d8bd222a25ef831da9e27ef4e0a807`.
Ricalcola binario/realpath, stdlib/ensurepip noti, wrapper/uv/config presence-hash
senza contenuti sensibili. Pip24.0 bundled esistente2110226byte SHA
`ba0d021a166865d2265246961bec0152ff124de910c5cc39f1156ce3fa7c69dc`, percorso
lib/python3.12/ensurepip/_bundled/pip-24.0-py3-none-any.whl sotto quel bootstrap.
Sola lettura stat/zip metadata/hash per proposta recupero, niente ensurepip/venv.
Nessuna esecuzione uv operativo/interprete managed/hook/prodotto/runner.

Nella request prepara argv/CWD/env/timeout giustificazioni dei passi futuri:
directory esclusive poi due helper separati; tool exec_command require_escalated
con review per ciascun comando operativo, login=False, nessun prefix ampio.
Driver ambiente noto: rimuove override Python/UV/PIP/SETUPTOOLS/proxy rilevanti,
registra soli nomi rimossi e configurazioni presence/hash, HOME invariato, PATH
bootstrap:/usr/bin:/bin. TMPDIR/cache del nuovo target, python-downloads=never,
non dipendere dal /tmp assente. Registra shell=False, close_fds e cwd assoluto,
hash orchestratore/allowlist/helper/processore e futura semantica outputtransizione.
Tool review futura negata = STOP, conservarla e riferirla; nessun nuovo tentativo
equivalente o cambio canale automatico. In questa chat niente tool require_escalated
per eseguire batch/acquisizioni: devi soltanto preparare comandi reviewabili.

Verifiche consentite: AST/JSON/TOML/hash/Git e piccoli test stdlib dell’helper con
stream HTTP **finti** in fixture nuove, zero socket/rete, senza import prodotto.
Usali soltanto per i rischi reali (redirect/cap+1/SHA mismatch/identità schema/
path preesistenti e conservazione parziali), non come prova HTTP o compatibilità.
Hash helper/test/fixture/report prima della request. Nessun pytest/suite legacy/
collection/runtime baseline: runner R futura separata, metadati HTTP preparatori
sono acquisizioni pubbliche esterne a R e non un test applicativo.

## Proposte rinviate e catena delle prove

Otto comandi s001 managed/lock/lock-check/deps/backend/build/sync/install tutti
rinviati. Non acquisire Python, wheel/sdist, tokenizer o backend; non generare lock
o S/B/I/E/probe. uv0.10.10/CPython3.12.13/setuptools84 fissi, no upgrade o rimozione
extra/no-build. Catalogo managed GNU generico/build20260310 resta candidato:
proposta statica per selezione uv/versione/variant/redirect asset/cap applicabile,
mai catalogo come prova di selezione runtime. Backend ignoto STOP prima hook.
Risoluzione universale e CPU/cu126 separati conservati, nessun taglio implicito.

Recupero baseline: dopo17JSON ricevuti, proposta distinta con URL/hash/size/tag
selezionati e bootstrap/pip24/tokenizer esatti, spazio/tempi/cap/argv; nuova
disposizione software e freeze prima ricostruzione. Vecchia venv non ricreata con
stessa identità fittizia. Newtarget e origine/RECORD/cache/R completa da provare;
originali sorgenti preservati, non moduli del prodotto P1–P3, per confronti baseline.
Negativi D2/D3, current↔S↔snapshot, flag/preflight/config-only e freschezza IDE
obbligatori futuri. Ogni cambiamento di input acquisito richiede label progressiva
prima della prova dipendente; S successivo al freeze, non artefatto futuro in esso.
Non correggere test/algoritmi legacy per ottenere PASS. Nessun GO codice da inventario.

## Consegna al supervisore

Prepara request.json **schema1**, run_id/revision/stage package/label
`impl-r001-stage-package-s002`, subphase PREPARATION_METADATA_INPUTS,
status **WAITING_FOR_STAGE_SNAPSHOT**, piano/arbitrato/contesto/completion/ricezione
esatti; elenco stabili+assenze, Git/base/branch/indice, host/toolchain/config/hash,
comandi/env/perimetro/costi/stop e lista artefatti esistenti root-relative confinati
alla run, senza symlink/.. o path assoluti. Usa lo schema request s001 come antecedente,
senza copiare ciecamente le sue assenze o2662record come hash disponibili.

Includi1269antecedenti invariati, stage s001 e suoi output reali preservati,
completion/proposte, questo prompt e contesto nuovo, disposizione/scope/ricezione/
fonti supervisore, tuoi helper/manifest/fixture/test/inventari/report di preparazione.
Report/checkpoint autore correnti sono mutabili: copia versionata degli ingressi
prima di includerli, non richiedere snapshot dei futuri output di ripresa.
La request stessa va aggiunta alla lista senza self-hash. Niente snapshot s002
o S/future receipt in artifacts; path host attuali distinti da artefatti copiati.
Tutti inventari identificano chiaramente host presenti, vecchi input solo storici,
output attesi s001 e nuove assenze s002. Più copie non sono equivalenze automatiche.

Consegna findings/costi/comandi/identità e report di preparazione sotto la tua
cartella, aggiorna **solo report e checkpoint propri**; preserva ogni FAIL/receipt
precedente. Proposta di changelog nel report, non modificare i sei metadata tracciati
comuni né STATE/HANDOVER/events/artifacts. Controlla link/JSON/stati/diff --check e
che prodotto/hash antecedenti invariati. Riscontri aggiornati fino alla consegna,
nessun software acquisito o processi lasciati attivi. Termine della chat con request
e checkpoint WAITING_FOR_STAGE_SNAPSHOT, poi nuova chat supervisore prompt05.
Non congelare da implementatore, non proseguire dopo invio come servizio continuo.

V1–V9 e due review codice/arbitrato finale ancora futuri; V7/V8 obbligatorie con
costo pesante distinto, V10/V11/pesi/font/inferenza non autorizzati. Nessun invio
di documenti o costo pagato/benchmark remoto, modifica host/sysctl/AppArmor/profili/
setuid/rete/daemon o altro progetto. Nessun commit/merge/push/promozione/deploy;
Git manuale dell’utente dopo GO finale. Temp/cache/venv ignorate non viaggiano
con Git: trasferire file e lavoro identificati, niente cleanup in questo mandato.
