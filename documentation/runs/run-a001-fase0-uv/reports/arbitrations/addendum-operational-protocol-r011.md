# R011 — recupero metadata e completamento lock/check nella stessa chat

2026-10-06, supervisore, seguito del pianoGO r003 e delle correzioni r009/r010.
Ricevuto42/report-r008: errore MAX_ARG_LEN corretto localmente, poi R/D4 PASS
con native uv nello stesso namespace; lock FAIL per cache incompleta, check
non eseguito. Non è un conflitto dimostrato con registry completo. S012 MATCH
all’ingresso,57file d’autore e759record cache ricalcolati, baseline preservata.

## Disposizione sulla proposta dell’autore

Non accolta come soluzione sufficiente la formula pip compile --no-deps:
nel sorgente uv0.10.10 visit_candidate richiede metadata soltanto in modalità
transitive (riga1669) e get_dependencies termina presto in modalità direct
(riga1808). Può alimentare index ma non i47metadata wheel mancanti. Help CLI
non dimostra questo comportamento. Nessun priming avviato in ricezione.

Ricevuto un mandato per risultato: **priming diagnostico nativo transitive
binary-only → audit/chiusura cache → lock universale offline/R/D4 → check**.
Il pip compile diagnostico NON è il grafo prodotto e il suo output non viene
promosso. Input iniziale20nomi e candidati scikit-learn1.9.0/narwhals2.26.0
servono a leggere metadata; non sono nuovi pin o constraints del progetto.
Configurazione TOML vuota esplicita/no-sources separa quel comando dal prodotto.
Global --no-build/only-binary sul priming impedisce backend sdist. L’eccezione
EbookLib osservata continua esclusivamente nel lock/check offline.

## Delega della chiusura e dei tentativi

Ammessi i18nuovi index già osservati e nomi dipendenti provati dal trace/native
metadata, o altri miss PyPI del medesimo grafo prodotto, entro **64nomi nuovi
complessivi** rispetto ai114 d’ingresso. Non richiedere altro mandato per un
nome/candidato/piattaforma diagnostica entro questi limiti. Fonte PyPI, pin,
indici CPU/cu126, requirementsPython, extra e grafo universale del prodotto
non cambiano. Niente requirements diagnostici promossi, nuovi override metadata,
lock scritto a mano o cache fabbricata. Per package con fonte dedicata non
sostituire la fonte con PyPI per far passare il lock.

Prima di ciascun lock offline registrare/sigillare l’insieme reale degli index
cache e la provenienza dei nuovi nomi; confrontarlo prima/dopo. Generare divieti
build da **tutti** i nomi reali salvo ebooklib, aggiungendo il progetto; conservare
Marker/setuptools e ogni divieto precedente. Il template iniziale132nomi non è
un gate che impone esattamente132, né un’allowlist native generale. Source/build/
Git cache restano prive di sorgenti/alberi, salvo marker tecnici già ammessi.
Un cambiamento inatteso durante il lock offline resta FAIL, non si normalizza.

Cache e lista derivata sono output dell’acquisizione autorizzata: hash/inventario,
sealed-inputs e actual argv vengono registrati prima del nuovo workload. Nessun
freeze supervisore intermedio per ogni dato acquisito; s013 congela gli input
stabili, la derivazione ammessa e la policy. La lista ricevuta resta immutabile
come evidenza di ciascun tentativo; non mutare i file scope/policy congelati.

R010 vale per errori CLI/launcher/reader, suffissi output esclusivi e timeout
entro gli intervalli ammessi. Metadata mancanti ulteriori entro la chiusura
ricevuta si recuperano nella stessa chat e si ripete il lock dopo diagnosi/gate.
Priming900s/lock900s/check120s cumulativi di workload per l’intero seguito;
extensions/guardie esterne rispettano deadline+180. Niente retry cieco, parziali
cancellati, sessioni duplicate o trasferimento di PASS fra input modificati.

## Rete, payload e risorse espliciti

Acquisizione native uv separata con HTTPS pubblico PyPI/files.pythonhosted.org,
env chiuso e nessun auth/proxy/config ereditato. Non è R offline PASS: quel
claim è riservato a lock/check sotto Firejail net=none/D4. Nessuna modifica ai
profili host, rete/daemon/socket host, privilegi o altri progetti. Prima dei
probe verificare startup/origine dei medesimi interpreti; zero import app/native.

La documentazione del sorgente pinned mostra fallback dalla lettura metadata
alla wheel intera se lo streaming non funziona. Lo scope ammette, per la sola
chiusura diagnostica, wheel necessarie a leggere metadata entro i caps: contare
body/copie/espansione/cache, nessuna installazione/ABI. Non dichiarare «solo
sidecar scaricati» senza osservazione. Vietati sdist/Git/backend, pesi/font,
modelli e archivi binari completi torch/nvidia/Marker/Surya/OpenCV. Redirect/URL
fuori host o payload fuori perimetro ferma l’acquisizione; monitor e trace non
sono whitelist di rete o quota fisica atomica. Se non osservabile, limite
esplicito, nessun PASS inventato. Non generare hash tramite download massivi.

Stima d’autore4MiB attività+2MiB registri non è un cap verificato. Ammessa
contingenza concreta: **tranche cumulativa estesa16→32MiB**, stesso
Hentry525762560byte;24MiB attività/8MiB registri,16MiB esterni. Costi precedenti
inclusi, nessun reset/nuova tranche32 oltre alla vecchia16. Pool1GiB/stop896MiB,
libero1GiB; H=max(logical,allocated),Delta=max(0,H-Hentry), gate
H+max(0,32MiB-Delta)+16MiB<896MiB e Delta<32MiB, caps attività/registri separati.
Monitor0,5s/gap target1s non atomicità; file32MiB/log1MiB/JSON8MiB. Body stimati,
wire non misurato; zero servizi a pagamento. Niente cleanup o ledger esterno.

## Consegna e limiti

Report-r009, completion s013, delivery/actual argv/hash/receipts di ogni tentativo,
manifest della cache acquisita e sealed names, origine endpoint/provenienza,
audit lock e check. Solo report/checkpoint dell’autore; stato comune al supervisore.
Se riuscito consegnare insieme richiesta concreta di promozione e S/B/I/E, senza
eseguirli implicitamente. Blocchi nuovi fuori perimetro/budget/isolamento o
conflitto del grafo realmente dimostrato tornano al supervisore, con prove/fix
concreto; non fermarsi per semplici ulteriori metadata entro l’autorizzazione.

Baseline/suite62pass5fail e perdite, s009 byteFAIL/lacune restano storici. V7/V8
pesanti distinti, V10/V11 esclusi, due review reali e arbitrato finale necessari.
Nessun GO finale/commit/merge/push/deploy/cleanup. Git manuale dell’utente.
