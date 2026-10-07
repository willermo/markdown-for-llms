# Implementatore r001 — Chiudere i rilievi delle due review

Prosegui nella stessa chat che ha consegnato report-r017. **Completa in un solo
giro i fix e le prove richieste dall'arbitrato**, risolvendo gli errori ordinari
nella stessa chat. Nessun nuovo piano, attesa freeze o consegna per un test rosso.
Repository `/home/davide/workarea/markdown-for-llms`, branch feature/run-a001-uv,
HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091; worktree autorizzato
modificato, indice vuoto. Non resettare, fare Git o modificare report storici.

## Autorità e ingressi

Leggi AGENTS, skill manage-implementation-run, protocollo e il tuo checkpoint;
poi sotto `temp/run-a001-fase0-uv/`:

- [Arbitrato implementazione r001](../arbitrations/arbitration-implementation-r001.md),
  autorità dei sei fix e delle verifiche sotto.
- `reviews/review-implementation-r001-chatgpt.md` e
  `reviews/review-implementation-r001-claude.md`, prove dei sei rilievi nei rispettivi
  `evidence/review-implementation-r001-*/`. Ora puoi leggere entrambi.
- `implementation/report-r017.md`, `evidence/implementation-r001/completion-r003/`
  delivery/final-checks/equivalence/matrix e input delle prove riusate.
- Piano r003 e arbitrato piano r003 (D2/P6/P7 in particolare); R016/R017/R018 e
  `evidence/supervisor-implementation-r001/final-core-resources-r001/authorized-scope.json`.
  In questa chat recupera le parti pertinenti; in chat fresca leggi il piano completo.

final-s001 era MATCH alla ricezione delle review; gli aggiornamenti documentali
del supervisore lo rendono storico. Non sovrascriverlo o aspettarne MATCH sul
nuovo worktree. Crea tu snapshot di esecuzione/S pertinenti, con label nuove,
senza checkpoint vivi o output futuri fra gli input di se stessi.

## Fix completi

### CLA-I001 — Test ordinari e run successive

Rendi esplicita la scelta della modalità standalone nel conftest/harness:
flag/env documentato e validato, senza inferire la modalità dall'assenza di temp
né degradare automaticamente una prova ufficiale. Le receipt standalone restano
standalone; non diventano prove ufficiali della catena a001.

Mantieni origine installata, -I/-B, confronto corrente↔S↔B/I, cache/Pandoc e
guardie di rete prima della collection. Generalizza snapshot_check al run_id
valido dichiarato, confinato al clone/run/snapshot corretto e verificato dal
tool di contesto; niente accettazione arbitraria di path/ID/SHA.

Dimostra C-fast reale con S standalone e receipt corrente dal percorso normale,
in una copia sintetica del clone **senza temp locale e senza snapshot di run**.
Usa risorse/interpreti già verificati e output propri entro ledger; se serve
Git alla copia, è ammesso init locale della sola copia sintetica senza commit,
remoti o mutazioni del repository condiviso. Non nascondere temp del clone vero.
Negativi: S stantio e receipt stantia/mismatched falliscono prima di collection
e import applicativi, senza sync/riparazione. Dimostra anche run_id diverso con
schema valido, confini invalidi respinti, official rifiuta standalone e official
positivo pertinente. Un mock isolato non sostituisce C-fast richiesto.

### CLA-I002 — Build Compose riproducibile dal repository

È scelta l'opzione di **riparare i due Compose di build**, conservando il piano.
Imposta rete build none, contesto usa e getta esplicito preparato da strumenti
versionati, additional_contexts della supply locale e riferimenti pinned.
Nessun binding image-source generato nel clone operativo né supply inclusa in
Git. I path relativi dei dati/runtime vanno risolti correttamente anche usando
il contesto preparato; non avviare servizi per verificarli.

Promuovi/adatta nel repository gli strumenti realmente necessari per preparare
supply APT/wheel e image-source/contesto. Sono permanenti, con CLI repo/input/output
e scope espliciti, senza dipendere da temp a001, HOME personale, stage vecchi o
helper non versionati. Il nome storico della directory diagnostici non obbliga
un refactoring estraneo. Conserva provenienza, URL/digest/dimensioni/ABI,
APT firmato e resolver nativo, backend84/EbookLib, lock e allowlist reali.
Riuso locale verifica tutto prima dell'uso; per un clone normale documenta la
preparazione completa con questi tool e acquisizione esplicita delle sole fonti
configurate. Non sostituire la procedura con una copia di input opachi della run.
Non scaricare nuovamente la supply già disponibile per dimostrare il fix.

Esegui una build CPU reale dal **Compose consegnato** sul contesto finale con
rete none e immagine identificata; un compose config o il vecchio buildx non
sono questa prova. Verifica entrambi i config CPU e override GPU solo config.
Per il secondo Compose è ammessa equivalenza puntuale del build risolto
(context, Dockerfile, network, args, additional_contexts, platform/target e input)
con quello realmente costruito; se differisce, costruiscilo e verifica il delta.
Conserva sentinelle sintetiche/contesto effettivamente filtrato, S/B/I builder e
runtime, isolamento/startup, inventari, digest e costi. Non usare up --build,
CMD normale, pesi/font, inferenza o GPU. Le vecchie V7/V8 restano prove di r005.

### CLA-I003 — C-docker permanente e realmente eseguito

Porta nel test/driver versionato i controlli del driver V8 reale, senza dipendenze
da helper in temp: container per ID/pull never/network none, nessun mount,
cap-drop ALL/no-new-privileges, healthcheck disabilitato, limiti memoria/pids,
filesystem/startup controllati a container fermo prima di Python e full I prima
degli import. Stdin fidato e nove sottocasi offline invariati, nessuna inferenza.
Conserva raccolta dei soli processi/container propri e receipt anche sui FAIL.
Esegui **C-docker versionato** sul risultato finale di CLA-I002 e registra comando,
esito/nodeid/receipt e input reali. Gestisci il preflight host senza renderlo
dipendente da uno snapshot storico; il daemon locale resta nello scope separato.

### CLA-I004 / CLA-I005 — Documentazione completa e durevole

Riconcilia singolarmente i74blocchi e9inline del README base: mantenuto/modificato/
rimosso/spostato, identità originale, destinazione valida o motivo specifico.
Recupera in docs la configurazione JSON (directories, validation_thresholds,
cleaning_settings, chunking, conversion_settings), scelta backend Marker/Pandoc,
multi-formato/polling e scenari ancora validi confrontando il codice attuale.
Non ripristinare comandi inesistenti, pseudocodice come operativo o promesse ML.

README/guide descrivono procedure e limiti durevoli; lo stato transitorio della
run rimane in documentation/temp gestito dal supervisore. Correggi la frase
AGENTS sulla guida legacy. Nessun GO/inferenza dichiarati. Termina questi edit
**prima** della canonica finale: README è input B e cambia il METADATA.

### GPT-I001 — Cleanup e diagnosi del launcher

Pulisci il path socket realmente creato, preserva l'errore originale e scrivi
receipt anche quando cleanup fallisce, distinguendo l'errore secondario.
Test mirati EACCES e timeout dopo bind: nessun falso PASS, errore originale
conservato e temporaneo proprio raccolto; negativo cleanup fallito con receipt
e segnalazione esplicita. Mantieni R/D4 e nessun fallback non confinato.
Esegui R reale pertinente con wrapper finale prima dei nuovi workload ufficiali.

## Verifiche e invalidazione

1. Prima del lavoro pesante misura/ammetti quote e scegli output compatti e
   timeout entro limiti. Definisci una matrice dei sei ID con fix/prova/risultato.
   Questa matrice è lavoro tuo, non una richiesta di approvazione preliminare.
2. Edit finali di README/input build → nuove sdist/wheel, B audit/hook nativi e
   installazione canonica. Ottieni S/B/I/E finali root/base-a/base-b/esterno/dev/API
   con stesso interprete nei figli. Nessun solo sync come attestazione.
3. Dopo conftest/diagnostici nuovi: C-fast ufficiale, standalone richiesto,
   packaging/API e discovery pertinenti, zero skip/collection nascosti; negativi
   pre-collection e R aggiornato. V1 config-only/flag/stale/restore e outer postcheck
   vanno rivalidati per i gate/build input toccati, seed attuale prima di R.
4. Nuova V7 Compose e C-docker/V8 sul risultato finale. Riuso di baseline/CLI/
   fedeltà e altre prove solo con equivalenza puntuale degli input consumati;
   nessuna ripetizione indiscriminata per il cambio di label o soli documenti.
   Nessun algoritmo cleaning/validation/chunking da correggere in questo fix.
5. Controlla link, help/comandi,74+9disposizioni, whitespace e stati. Lascia A7/
   review/arbitrato al supervisore, V10/V11 e inferenza escluse. Conserva i FAIL.

## Risorse, autonomia e consegna

Quota core **304MiB cumulativi**, Hentry553541632, incremento318767104,
pool1073741824/stop939524096/riserva16777216byte invariati. Radici ed esclusioni
esatte dello scope R018; non spostare core sotto Docker per eludere la quota.
Ingresso tempo core3940.5060664880657s su7200; spazio ingresso va ricalcolato
includendo review/supervisione, non riusare il residuo storico21MiB.
Docker16GiB/7200s/rete1GiB, ingresso568.7285651748534s/storageupper11247782439/
networkupper496040215byte; wire non misurati. Prima del nuovo Docker conta anche
i due record cache-only dichiarati da Claude con stima conservativa identificata;
non escluderli solo perché piccoli. Nessun reset/cleanup globale/prune.

Riuso supply/basi locali, preparazione distinta da suite, monitor/caps attivi e
ammissione prima del lavoro. Timeout/log/output/launcher e soluzioni reversibili
entro requisiti e costi sono delegati per obiettivo. EPERM/EACCES: diagnosi e
approvazione strumenti per azione autorizzata se necessaria/disponibile,
mantenendo R; solo rifiuto effettivo o nuovo confine sostanziale ferma quel lavoro.
Solo quota/confine reale necessario oltre mandato richiede una proposta unica
consolidata; continua intanto i fix/prove indipendenti autorizzati.

Consegna **`implementation/report-r018.md`**, evidenze nuove sotto
`evidence/implementation-r001/fix-review-r001/` (Docker nei suoi soli root dedicati),
matrice chiusura sei ID, manifest finale/riusi/invalidation/costi e checkpoint
`handovers/implementation-r001.md` aggiornato. Non scrivere registri condivisi,
arbitrati o review. Non fermarti a patch pronta, test rosso o snapshot da congelare.
Alla consegna completa il supervisore prepara due nuove review mirate e arbitra;
Git manuale dell'utente solo dopo GO finale.
