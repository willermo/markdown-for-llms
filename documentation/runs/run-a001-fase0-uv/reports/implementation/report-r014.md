# Implementazione r001 — mandato49, approvazioni reali e fix RECORD

2026-10-06. **WAITING_FOR_STAGE_SNAPSHOT product-s018**, owner supervisore.
Due richieste concrete `exec_command require_escalated` accettate ed eseguite:
S2 e driver dei21 passi successivi. Nessun rifiuto auto-review. Sul freeze s017
ottenuti5 PASS (S, build nativa, audit B, sync root, canonica root), poi I root
exit2/FAIL per uv_cache.json non riconosciuto dal reader;16 passi NOT_RUN.
Difetto corretto qui con13 verifiche locali PASS. Modifica a un input della
prova: s017 ora STALE, nessuna ulteriore prova ufficiale attribuita a quel freeze.
Input completi della ripetizione già raggruppati; non è richiesta di permesso
per un adapter/argv o una nuova pianificazione.

[Delivery22esiti — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[request s018 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[regressione reader — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[input nuovi — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).

Ingresso R015/ricezione supervisore verificati. S017293347byte
SHA0949ad6715795fe8ee91c387a6e2d1714f2958ac8dd7530a3f81226a4e5216bc,
worktree72951e483c8e9509fd09462591937650dd25f133005d26e0cd4f15f9448c05ed;
127file/938artifact. Branch feature/run-a001-uv, HEAD/dev/base66ba822,
indice vuoto, scope63176/SHA3fd09c86e4cd4d124c3df035cf84ee102b97fee7c372a5e27689d8463134e4c7.
GO del solo piano r003, NO_GO storici, nessun GO/review finale.

Adapter operativo proprio resume_product_s017.py, bytes/digest nella delivery:
carica launcher/template originali identificati, li preserva e seleziona s017
in memoria per snapshot_check, wrapper e argv S. Environment pubblico puro,
os.environ svuotato e sostituito prima dei subprocess dei diagnostici; shell=False,
close_fds=True. Startup managed/bootstrap e config guardati prima degli interpreti;
target/startup reali per ogni operazione. Contabilità autorevole111.10300820460543s
più120s preparazione conservativa, poi intervalli adapter non sovrapposti.
Ogni monitor limita realmente il wrapper al residuo, figlio<=900 e outer+180;
le stime interne del dispatcher congelato restano storiche e NON il gate.
argv modificato al limite reale in authoritative-invocation.json.

Prima richiesta esatta: managed python -I -B adapter --operation S-frozen-product
--attempt2 (argv separati --attempt e 2), CWD radice; session26393, exit0.
Seconda: medesimo adapter --remaining-sequence, session79280, exit1 al prerequisito
I. Nessun prefix_rule generico. tool-approval-observations.json conserva comandi,
scope e osservazione del tool; testo interno del reviewer non esposto dal sistema.
Approvazione osservata perché lo strumento ha avviato la sessione richiesta.
Socket sintetici solo nella run, daemon probe connect/close senza send/recv
operativo; nessuna modifica host/sudo/setuid/profili o rete/download. Ogni
workload ha nuovi R/D4 e netns figli verificati dentro Firejail net=none;
approvazione distinta dalle prove di isolamento. Processi raccolti dopo ogni
tentativo: nessun sopravvissuto osservato. Nessun app/CLI/suite/dev/API/ML.

S ufficiale riferisce s017 e805 input reali; ID/file SHA nella delivery.
B nativa offline uv no-build-isolation/constraints84/backend ricevuto. Startup
e file backend guardati, import backend84 e get_requires sdist/wheel reali prima di B:
entrambi vuoti. Raw cmdline nel medesimo netns osservano setuptools.build_meta
build_sdist e build_wheel. uv default produce wheel dalla sdist; veri nomi/hash
derivati dall'output. Audit B tar/ZIP/metadata/RECORD/diecihash PASS prima install.
Sync locked root no-install-project/no-build PASS, canonica no-deps/no-build PASS.
I FAIL è conservato: RECORD installato uv aggiunge metadata uv_cache.json.

Decisione/fix: correggere verify_distribution.py, non rimuovere uv_cache.json
dal target né saltare RECORD. Il reader ammette solo extra installer nominati,
legge e verifica hash/dimensioni di uv_cache.json, INSTALLER/REQUESTED/direct_url,
cinque script generati e pyc previsti. Script vincolati al bin del prefix;
altri extra a site-packages, niente symlink/file speciali, limiti di byte.
RECORD duplicato/incompleto/ignoto rifiutato; pyc con campi vuoti ammessi da
PEP376 ma bytes effettivi inclusi nei descriptor. La allowlist ZIP non cambia.
Ora i campi precedentemente saltati vengono verificati, anziché ignorati.

Regressione locale stdlib, senza startup del target/app: RECORD root reale
completo PASS, confronto dei file wheel installati byte identico alla canonica;
12 negativi su hash/size/cache mancante/duplicati/path sconosciuti o evasivi/
script symlink o mancante e extra installer nella wheel. Non è I ufficiale:
nessuna validazione corrente/S/snapshot del reader corretto su stage STALE.
Diagnostico originale preservato verify_distribution-frozen-s017.py.
Ultimo MATCH s017 prima del fix in before-diagnostic-fix.json; final verify
STALE atteso documenta il solo diagnostico cambiato, non un MATCH nuovo.
Dieci moduli, lock, metadata prodotto, backend e input runtime invariati.

Request s018 conserva938 artifact ricevuti più output reali e fix; checkpoint
vivo escluso. Nuova inventory805 modifica il solo descriptor del diagnostico;
805 bytes/hash attuali verificati. Launcher/template originali intatti.
Adapter s018 identificato importa quello operativo s017 immutato e usa nuova
directory evidence, inventory/S/dist/I/export e base-a/base-b in product-sbi-r002.
Prima del freeze manca s018: nessun --operation ufficiale eseguito. Guardie e
semantica R/D4/audit restano originali; copierà solo template storici identificati.
Root .venv già presente: nuova sequenza usa fresh sync/reinstall; nessun falso
vincolo di assenza. Altri target ed esterno assenti. Nessun overwrite S/B/receipt.

Contabilità nuova prima della consegna487.34544921084307s, residuo6712.654550789157s:
111.10300820460543 pregresso +120 preparazione adapter + intervalli delle6
invocazioni + verifica pre-fix + regressione reader +120 allowance fix/ingressi
+60 allowance consegna. Preparazione/controlli inclusi nelle allowance, monitor
annidati non aggiunti; dettaglio nella delivery. Entrypoint s018 impone già tale
cumulativo e non resetta la quota. Pool112MiB/Hentry553541632 invariato, formule
lstat nofollow/free/cap rispettate; ledger/gap massimi effettivi nella delivery.
Monitor periodico0.5s/gap target1s, nessuna garanzia atomica; paid0/download0.

Errore locale nel primo collector di consegna: assumeva un campo bytes negli
entry dello snapshot, che identificano i file tramite SHA. Versione fallita e
causa preservate; fix legge SHA e confronta bytes quando presente, senza cambiare
la verifica ufficiale. Collector ripetuto PASS, costo incluso nell'allowance60s.

Report/checkpoint autore aggiornati, stato comune/events/arbitrati/changelog e
snapshot al supervisore. git diff --check e collegamenti/identità verificati.
Prossimo ruolo: supervisore riceve request ed esegue freeze ufficiale s018,
poi autore ripete l'intera22 sequenza con nuovi S/B/I/E. R015/mandato49 richiedono
quel freeze quando cambia un diagnostico della prova: nessun auto-freeze o
trasferimento dei5 PASS a input corretti. Prove V1 config-only/rebuild/stale,
V3–V9/import/fiveCLI/fasi/dev/API/suite/comparativi/negativi aperte; V7/V8 a costi
distinti, V10/V11 escluse. Review ChatGPT/Claude nuove e GO finale futuri.
ABI managedLinuxx86_64, baseline62/5/perdite e FAIL storici preservati.
Nessun agente/staging/commit/merge/push/promozione/deploy/cleanup globale.
