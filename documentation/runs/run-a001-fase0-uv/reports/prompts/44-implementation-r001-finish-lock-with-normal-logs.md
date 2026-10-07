# Implementazione r001 — completare metadata, lock e check con log normali

Agisci esclusivamente come **implementatore** nel repository
`/home/davide/workarea/markdown-for-llms`. Puoi proseguire nella stessa chat.
Obiettivo: completare la chiusura metadata già ricevuta, ottenere il lock
universale offline e verificarlo. **Esegui e correggi nello stesso mandato**;
nessun passaggio di sola preparazione per errori ordinari. Nessuna delega.

## Autorità e input

Leggi AGENTS, skill manage-implementation-run e protocollo; STATE/HANDOVER comuni,
checkpoint `handovers/implementation-r001.md`, report-r009 e checkpoint supervisore
`handovers/supervisor-normal-logs-r001.md`. I path seguenti sono relativi a
`temp/run-a001-fase0-uv/`:

- `arbitrations/addendum-operational-protocol-r012.md`: disposizione corrente;
  r010/r011 restano applicabili per autonomia, rete e source-policy;
- `implementation/stages/impl-r001-stage-package-s014/request.json`;
- `evidence/supervisor-implementation-r001/impl-r001-stage-package-s014/`:
  authorized-scope.json, reception.json, transition.json, checks.json,
  pre-freeze-checks.json, freeze-verify.json, metadata-prime.in;
- `snapshots/impl-r001-stage-package-s014.json`;
- evidenze autore `evidence/implementation-r001/resume-package-s013/`,
  in particolare final-checks/sealed-cache-r003/cache-acquisition-audit/next-gate-request.

Piano r003 SHA `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`,
arbitrato r003 `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`.
GO sul piano nei limiti, nessun GO finale; NO_GO storici conservati.
Scope completo è fonte di argv/env/cwd/input/policy e relativi limiti: leggilo,
non ricostruire i valori dal riepilogo.

Manifest 140730 byte, SHA `902b0f97e5992f41fa0fc8b46da4ffa4f617d9971affd7de29c4093a122942ee`;
worktree `38658736cf972ff9c8591076a3add3a157b5e11cce05d7e6ddcae0de18a1024e`, 123 file/430 artefatti.
Request 51890 byte, SHA `df1672c7fa42a4b24aeffae2e289c78b39a80b1010a9606fadde8342db2eaa48`;
scope 57532 byte, SHA `a4cba091ef4b2bf162e16a124ef8980a61c4b64daa11bdc9d313a94d0553e95e`.
Ricalcola byte/hash e da radice esegui prima e dopo le operazioni:
`python3 -B scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-package-s014`.
Mismatch resta FAIL; niente auto-freeze o correzione dei gate congelati.
S013 è storico per i sette delta documentali dichiarati in transition; tutti
i suoi artefatti e gli input prodotto/helper restano intatti. Non esigere
MATCH del vecchio worktree contro la governance nuova.
Git feature/run-a001-uv, HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto e modifiche preesistenti identificate. Nessuna modifica tracciata
in questa tranche, nessun commit/merge/push/deploy/cleanup.

## Correzione ricevuta e autonomia

43 ha acquisito139 nomi/25nuovi,844record cache; tre priming exit0 e R/D4 PASS.
Lock non prodotto: ultimo stderr1164160byte oltre1MiB e registri oltre il vecchio
sottocap8MiB. Receipt esterna ultima assente; parziali/FAIL/log intatti.
`timed_out:true` interno indica tempo **oppure log**, non prova che900s siano
esauriti: durata0,206s. Conserva il dato grezzo e descrivi la causa osservata.

**Usa livello uv normale**: niente --verbose/-v/-vv né --quiet/-q. Il wrapper
interno ha cap fisso1MiB; non basta alzare un monitor esterno. Nessun taglio
storico, soppressione errori o falsificazione ricevute. Scope contiene template
senza verbose, inline compatto sotto4128byte e argv espanso corrispondente.

Adatta/correggi i tuoi launcher, reader e monitor non congelati per s014:
nuovi path esclusivi, consumo cumulativo e pool condiviso. Non eseguire invariati
strumenti precedenti che abbiano hardcoded s013,8MiB o --verbose. Sintassi,
output/encoding, livello log, timeout entro residui, ricette execve equivalenti
e miss della chiusura ricevuta si correggono autonomamente; preserva esiti
invalidati e registra input/comandi reali. I template di lancio non vietano
questi adattamenti. Nessuna modifica a wrapper/runner, snapshot, scope, baseline,
config/env/fonti/pin/grafo del prodotto. Retry solo dopo diagnosi e raccolta dei
processi propri, nessun retry cieco. Verifica staticamente i nuovi launcher
prima del primo workload senza inventare un nuovo ciclo di preparazione.

## Budget unico e tempi residui

**Stesso totale32MiB già autorizzato**, non32 nuovi: Hentry525762560byte,
H=max(logical,allocated), Delta=max(0,H-Hentry), Delta<33554432 e
H+max(0,33554432-Delta)+16777216<939524096. Conta run viva e .venv-python,
metadati directory/link con lstat senza seguirli; include costi storici,
launcher, snapshot e questa consegna. Nessun cleanup/reset/spostamento fuori ledger.

I vecchi24MiB attività/8MiB registri sono sostituiti dal pool condiviso;
i due campi legacy32MiB sono sottoinsiemi **non additivi**, nessuna riserva
interna24MiB o nuovo sottocap12MiB. Circa11MiB residui dopo il freeze:
misura valori reali prima e durante. Pool1GiB/stop896MiB/libero1GiB/riserva
esterna16MiB invariati. File32MiB, JSON8MiB, stream1MiB; monitor0,5s/gap target1s
non quota atomica. Usa inventari compressi e riferimenti a sigilli invariati;
non duplicare a ogni tentativo tutta la cache/storia. Conserva argv/exit/log reali.

Tempi **cumulativi residui** metadata898,0215938149486s,
lock899,488482683897s, check120s. Sottrai il consumo reale di ogni tentativo;
non900s nuovi a ogni label. Timeout figlio entro residuo, esterno=figlio+180s;
preflight/postcheck separati, raccogli tutte le sessioni e i gruppi propri.

## Esecuzione accorpata

1. Controlla input/provenienza/startup/origine/interpreti/config/baseline protetti
   secondo scope; misura ledger. Crea esclusivamente nuove evidenze
   `evidence/implementation-r001/resume-package-s014/`. Riusa il work metadata
   esistente, non ricrearlo né pulirlo. Output diagnostico nuovo come in scope.
2. Priming pubblico nativo uv0.10.10, managed CPython3.12.13: seed congelato
   fonttools[woff]==4.66.1, brotli/brotlicffi/zopfli, universale/transitive,
   **--only-binary=:all:**, no-sources, config vuota esplicita, no credenziali/
   keyring/proxy. Non aggiungere --no-build (incompatibile con only-binary nella
   CLI osservata), --no-deps o override prodotto. Il seed è diagnostico,
   non nuovo extra/pin prodotto; fonttools[woff] non autorizza asset font.
   Usa argv strutturati shell=False/close_fds=True e env chiuso esatto dello
   scope (19variabili), senza merge con os.environ. Zero install/backend/sdist/Git/
   pesi/font/archivi ML completi; limiti fallback metadata di r011 invariati.
3. Audit e sigillo cache/provenienza. Nomi/miss derivati dello stesso grafo
   sono già ricevuti entro64aggiunte dai114iniziali: i25acquisiti contano,
   massimo39ulteriori. Nessun handoff per ogni metadata mancante. Preserva
   marker tecnici, verifica bucket sorgente/build/Git e assenza nuovi payload
   vietati; non forgiare cache. Eventuali nuovi seed/varianti diagnostiche
   identificati con hash e motivazione, senza cambiare progetto/constraint.
4. Deriva --no-build-package per **tutti** i nomi reali sigillati salvo l'unica
   eccezione EbookLib0.18 già ricevuta. Lancia lock universale originale offline
   sotto wrapper/Firejail net=none: R completo e D4 reali sullo stesso processo,
   target e env R chiusi dello scope, inline compatto/argv espanso equivalenti.
   Prima di ogni workload registra input effettivi, cache e divieti derivati.
   Se manca ancora metadata ammessa, diagnostica→priming ammesso→nuovo sigillo→
   lock nella stessa chat, entro costi/tempi/nomi residui. Non restringere il
   grafo o cambiare pin/sourcepolicy per ottenere PASS. R43 non vale per R nuovo.
5. Solo dopo lock exit0 audit di lock/provenienza/Ebook metadata/grafo richiesto,
   poi lock --check offline con nuovo R/D4 e hash lock intatto. Nessuna
   promozione/installazione/wheel/S-B-I-E implicita in questa tranche.

HTTPS priming è distinto dall'offline R: raw URL/body/wire/header/redirect
non esposti dal log normale restano NOT_MEASURED, non attestazioni inventate
né quota di rete atomica. Acquisizione solo PyPI/file pubblici secondo r011;
nessun invio documenti o fallback remoto. Nessuna modifica host/privilegi/rete/
profili/socket, nessun kill di processi altrui. Superamento del pool globale,
nuove fonti/backend/perimetro, input protetti alterati, impossibilità R/D4,
rifiuto sandbox o conflitto dimostrato richiedono arresto del lavoro dipendente
con consegna concreta. Un normale errore del proprio strumento resta fix locale.

## Consegna

Scrivi delivery nelle nuove evidenze, `implementation/report-r010.md` e
`implementation/stages/impl-r001-stage-package-s014/completion-r001.json`, legati agli input reali
request/scope/manifest; esiti separati priming/R/lock/check, anche NOT_EXECUTED,
cache/provenienza, residui tempi/storage, processi e limiti. Aggiorna soltanto
report/checkpoint implementatore, preservandone gli ingressi con copie mirate;
nessun registro comune/changelog/auto-snapshot. Stato conclusivo
**WAITING_FOR_SUPERVISOR_RECEPTION**, prossimo supervisore05+r012.
Se lock/check riusciti, consegna insieme richiesta concreta per promozione e
S-B-I-E con comando/input/output/costi; non un'altra preparazione astratta.

Baseline originale1892file/211directory, suite storica62pass5fail/perdite,
s009byteFAIL/lacune e tutti i precedenti FAIL preservati. Nessun PASS ABI/V0
nuovo/GO prodotto. S-B-I-E/V1–V9, V7/V8 pesanti distinti e due review reali
ChatGPT/Claude con arbitrato finale restano da completare; V10/V11 esclusi.
Git manuale dell'utente. Non attendere servizi automatici dopo la consegna.
