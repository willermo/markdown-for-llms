# Disposizione utente — dati operativi della run nel temp del progetto

2026-10-04 Europe/Rome, supervisore Codex/OpenAI. L’utente dichiara di aver
liberato /tmp e di avere probabilmente eliminato la baseline insieme a dati
di altri progetti. È **causa probabile riferita**, non cancellazione dimostrata.
L’assenza della baseline rimane un’osservazione reale; confronti dipendenti IMPEDITI.
Il controllo statvfs conferma capacità /tmp4294967296byte (4GiB); temp del clone
appartiene al filesystem device66306 di capacità501809635328byte, con circa180GiB
disponibili al controllo. Dati puntuali e data in reception.json, non una garanzia
di spazio futuro né un nuovo budget di acquisizione.

L’utente chiede di usare una posizione adeguata dentro temp del progetto per i
dati generati e di pulire periodicamente ciò che non serve più. **Disposizione
adottata per nuovi dati operativi della run**, radice:
temp/run-a001-fase0-uv/work/. Sottodirectory separate e progressive, nessun riuso
dei path persi o spostamento automatico di output storici.

| Uso | Percorso assegnato dalla radice del repository |
| --- | --- |
| Metadata package s002 | temp/run-a001-fase0-uv/work/package-s002/sidecars e baseline-json |
| TMPDIR/cache operative s002 | temp/run-a001-fase0-uv/work/package-s002/tmp e uv-cache |
| Baseline futura ricostruita | temp/run-a001-fase0-uv/work/baseline-recovery-r001/venv,workspace,tmp,uv-cache,tiktoken-cache |
| Stage successivi | work con sottodirectory del proprio stage/label, assegnata prima del freeze |
| Evidenze e contesto | evidence/,snapshots/,handovers/ della run come già previsto |

temp è già ignorata: nessuna modifica .gitignore. Gli ambienti canonici del
prodotto .venv/.venv-python restano quelli del piano; la presente disposizione
riguarda cache, copie, venv baseline e artefatti operativi della run. Non modifica
algoritmi, toolchain, scope HTTP, cap o autorizzazioni software. TMPDIR dev’essere
coerente nel driver/padre/figlio e nel futuro runner, non un override host globale.
Prima delle prove: path reali/non symlink/assenza iniziale, spazio e device reali,
nuova request/freeze/S secondo D1; eventuali helper/argv/config modificati hanno
hash nuovi. Nessun PASS ereditato dal solo cambio di directory.

Prompt17 e next-scope originale sono congelati e **non riscritti**.
storage-scope-r001.json è una nuova versione del perimetro effettivo con gli stessi
22URL,5SHA sidecar,17pin/513hash/cap e limiti; cambia solo destinazioni di lavoro
e riferimento al contesto aggiornato. Addendum17a integra la chat implementatrice
già in avvio: la sua effettiva lettura non è assunta. Se prepara i vecchi path,
il supervisore conserva la consegna e dispone correzione versionata prima
di qualunque freeze/esecuzione. Nessuna altra chat contattata da strumenti.

In questo handover si aggiunge al changelog una voce di storage/passaggio:
il contesto prompt17 diventa storico per il solo CHANGELOG,1343artefatti invariati;
prima dello snapshot nuovo si registra la transizione. Il prossimo supervisore
distingue questa modifica documentale da eventuali divergenze tecniche nuove.
Stato/checkpoint correnti non sostituiscono report autore immutabili o l’identità
storica s001. Non inventare nuova request/preparazione conclusa/snapshot s002.

Nessuna pulizia o copia/move di directory operative ora. La pulizia periodica
richiede inventario selettivo della sola run, verifica di conservazione/archivio
delle evidenze necessarie e aggiornamento dei riferimenti; non cancellare input
ancora necessari o directory di altri progetti. Git non salva temp/cache/venv.
Costi software/ML, V7/V8 e criteri delle prove rimangono quelli già disposti;
V10/V11/pesi/font/inferenza esclusi. Commit/merge/push/promozione manuali utente.
