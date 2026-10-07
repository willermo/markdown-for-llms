# Addendum r009 — correggere la policy di selezione delle sorgenti nel lock

2026-10-06. Supervisore; mandato40 ricevuto. Stesso obiettivo36, piano/arbitrato
r003 immutati. Errata operativa di r005/r008, nessun GO finale.

## Esito e attribuzione

Lock offline s010 realmente FAIL/exit1, nessun lock; online/check non eseguiti
correttamente. Freeze MATCH all'ingresso, startup pre/post e preservazione verificati.
Il supervisore riconosce errata la premessa del mandato40: dependency-metadata
evita la build per ottenere metadati, ma non rende una sdist selezionabile con
il filtro globale no-build. Il FAIL non è imputato all'implementatore o al TOML.

Sorgente **uv0.10.10** verificato: version_map.rs521–532 rifiuta la source dist
con IncompatibleSource::NoBuild prima del consumo dei metadati. Non consulta la
cache wheel in quel ramo. Il successivo distribution_database.rs543–568 restituisce
invece i metadati dichiarati prima di download_and_build_metadata; resolve_revision
in source/mod.rs2015–2026 ritorna immediatamente per fonti registry non Git.
Confermati sorgenti pubblici pinned, non promessa derivata dal solo help.

Respingo la nuova proposta di wheel-cache registry/backend HTTPS: non corregge
quel filtro del resolver e aggiunge un'altra build/eccezione rete. Proposte r001
s009/s010 restano intatte/NOT_AUTHORIZED; nessun prime o installazione.

## Eccezione tecnica circoscritta e controlli sostitutivi

Per **questo lock e check offline** si omette il flag globale `--no-build` e
si usa `--no-build-package` per **ogni nome presente nei simple index della
cache congelata, eccetto ebooklib**, includendo esplicitamente marker-pdf,
setuptools e markdown-for-llms. Lista concreta/exhaustiva dell'ingresso nel
nuovo freeze; non è una allowlist offerta da uv. Nessuna build è autorizzata.

La sola source distribution resa eleggibile è EbookLib0.18 nel grafo corrente,
che ha i metadati osservati dichiarati/versionati. P1 del piano permette sdist
identificate e backend vincolati: il filtro globale era un'ammissione operativa
aggiunta dal supervisore in r005, non un requisito di flag del piano. L'eccezione
cambia soltanto la selezione nel lock di questa copia; i comandi futuri di
installazione delle wheel conservano i loro no-build, salvo nuovi mandati.

Il controllo non è affidato alla sola denylist: prima di ciascun avvio si verifica
che **tutti** gli index-cache names siano quelli congelati, senza nuove entry;
sdists-v9 contiene soltanto marker tecnici .git/.gitignore, nessuna fonte/build;
git-v0/builds-v0 non contengono sorgenti o build riusabili. Fonte EBook registry,
range richiesto>=0.18,<0.19 e metadati esatti0.18; nessun URL/Git/path sostitutivo.
Index sconosciuto non è disponibile offline; nessuna acquisizione di sdist può
avvenire. Input/identità della cache al momento d'ingresso sono ricevuti e devono
restare coerenti; un nuovo nome/source-cache ⇒ STOP, non ampliare la lista a runtime.

Entrambi i comandi passano **R → uv nello stesso namespace Firejail net=none/D4**
del wrapper tracciato, con target/argv/environment congelati. Env R16 chiavi e
nativo19 concordanti, inline execve già sperimentato; nessuna modifica del wrapper.
Zero rete, backend/hook/build/supply/install/import app. Il vincolo resta niente
esecuzione backend durante il lock: se trace/cache/processi mostrano un hook,
esito FAIL e STOP, non PASS sulla sola exit0. I probe del managed noto sono ammessi.
Nessuna modifica host/rete/daemon/privilegi/sysctl/AppArmor/profili persistenti.

È una garanzia **dipendente dagli input offline ricevuti**, non il claim che una
denylist di uv rappresenti un'allowlist generale. Non estendere questo comando
alla rete: non protegge backend di nomi nuovi. Cache insufficiente o altro source
non ricevuto ⇒ consegna concreta del blocco. Nessun ramo HTTPS in questa tranche.

## Mandato e costi

S011: due operazioni pronte, lock offline con policy ricevuta; dopo exit0 e audit
del lock reale, check offline con identici input/policy e SHA lock invariato.
Stessa copia16input di s010, nessuna nuova preparazione o ripetizione baseline/
build EbookLib. Nuovi tmp/config/output esclusivi, cache originale, vecchi log intatti.
Fix dei launcher/reader non congelati nella stessa chat; niente auto-freeze o
input congelati modificati. Errori ordinari non riaprono la pianificazione.

Nuova tranche **16MiB incrementali:8 attività/cache/R/output,8 registri**,16MiB
esterni. Ledger run+.venv-python max1GiB/stop896MiB/libero1GiB; dir/link inclusi
senza seguirli, quota precedente conclusa. Monitor0,5s/gap target1s non quota
atomica;log1MiB/stream,JSON8MiB,file32MiB. Lock child900s/outer1080s,
check120s/outer300s; sessioni proprie raccolte. Zero nuovi body/costi remoti,
cleanup per passare vietato, rifiuto sandbox IMPEDITA con ragione autentica.

Consegna lock/check/fonti/grafo/hash reali o causa puntuale, report nuovo r007 e
richiesta concreta dei successivi S/B/I/E/costi. Nessun trasferimento di PASS al
prodotto: il lock resta nella copia, root invariato, migrazione non convalidata.
S009 byteFAIL/pre-bootstrap mancante preservati; baseline originale caratterizzata
e suite62pass5fail acquisita, non rifatta. V7/V8 e doppie review/arbitrato finale
obbligatori; V10/V11 esclusi. Git manuale, nessun commit/merge/push/deploy/cleanup.

Fonti pinned: [selezione](https://github.com/astral-sh/uv/blob/0.10.10/crates/uv-resolver/src/version_map.rs#L521),
[metadati](https://github.com/astral-sh/uv/blob/0.10.10/crates/uv-distribution/src/distribution_database.rs#L543),
[fonti registry](https://github.com/astral-sh/uv/blob/0.10.10/crates/uv-distribution/src/source/mod.rs#L2015).
Ricevute/cache/esiti nella ricezione package-s010-reception-r001.
