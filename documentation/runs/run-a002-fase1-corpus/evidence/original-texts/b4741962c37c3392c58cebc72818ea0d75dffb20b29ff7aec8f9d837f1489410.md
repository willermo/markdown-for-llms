# Fix locali del piano a002 r001 — errata da completare

Agisci esclusivamente come **autore delle errata locali del piano** per
`run-a002-fase1-corpus`, proseguendo la chat pianificatrice se disponibile.
Il supervisore ha ricevuto entrambe le review e arbitrato **NO_GO locale**.
Il risultato richiesto è un addendum mirato, autosufficiente e verificabile,
non un nuovo piano generale né implementazione anticipata.
Usa la skill manage-implementation-run e il protocollo della run.

## Ingresso autorevole e identità

Leggi AGENTS, temp/HANDOVER, la sola sezione corrente di PROJECT-CONTEXT,
STATE/brief/HANDOVER della run, questo prompt e il checkpoint corrente
`handovers/supervisor-plan-arbitration-r001.md`. Poi leggi:

- `plans/plan-r001.md`, SHA256
  `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`;
- `arbitrations/arbitration-plan-r001.md` integrale: decisioni e disposizioni
  E01–E08 prevalgono nelle sole sezioni indicate;
- `reviews/review-plan-r001-chatgpt.md`, `reviews/review-plan-r001-claude.md`
  e il checkpoint originario `handovers/planning-r001.md`;
- `notes/user-gold-visual-verifier-2026-10-08.md` e le evidenze pertinenti in
  `evidence/supervisor-plan-arbitration-r001/`;
- protocollo run-lifecycle, skill manage-implementation-run e
  verify-conversion-fidelity, ADR0001, workflow benchmark e guida uv/reference
  soltanto per i contratti interessati dai fix.

Non caricare tutta a001, non riscrivere il piano originale, i report review,
gli snapshot, gli arbitrati o il checkpoint del supervisore. Non impersonare
reviewer o supervisore. Puoi leggere i due report ora: questo è il ruolo autore
dei fix dopo l'arbitrato, non una review indipendente del medesimo giro.

Base/HEAD/dev verificati dal supervisore:
`80c31da46e5caab990d3596c79c44e524ec1dc95`, branch feature/run-a002-corpus,
indice vuoto e delta documentali ereditati. Verifica lo stato attuale prima
di scrivere; nessun reset, commit, merge, push, pulizia o Git di scrittura.

Lo snapshot plan-r001 è storico delle review e resta immutato. Gli aggiornamenti
documentali dell'arbitrato ne cambiano l'impronta generale; la transizione è
registrata, senza attribuire STALE a un mutamento del piano o a review invalide.
Usa il nuovo snapshot tecnico **plan-fix-context-r001**, che identifica gli
input effettivi del mandato, prima e dopo il tuo lavoro:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label plan-fix-context-r001
```

Le errata/checkpoint/evidenze proprie sono nuovi output sotto temp e non
modificano gli input congelati. Se un input protetto è difforme, conserva
diagnosi e continua le sole correzioni indipendenti; non ricreare snapshot
per nascondere il delta. Identifica provider/modello effettivi, non inventarli.

## Risultato completo delle correzioni

Scrivi `plans/errata-plan-r001-r001.md` con un'intestazione che identifica il
piano originale, l'arbitrato e questo prompt. Organizza per E01–E08, indicando
sezioni/righe sostituite, specifica risultante completa, ragione, requisiti
preservati e prove future di chiusura. Nessun segnaposto per i quattro blocchi.
Il corpus resta 10 documenti/9 pagine PDF, le classi e il risultato del brief invariati;
PDF/OCR e compilazione TeX non si trasformano in prove già misurate.

1. **E01 / CLA-P001:** definisci grammatica/schema strutturale per il
   sottoinsieme matematico del corpus, origine indipendente del gold,
   normalizzazioni contestuali e distinta diagnostica raw. Fornisci matrice
   concreta di formule in grafie alternative con struttura attesa, più
   negativi su segno/indice/argomento/righe/celle che devono restare FAIL.
   Non basta scrivere «canonicalizzare»: elenca regole e limiti/UNAVAILABLE.
   Niente CAS, riordino algebrico o regex globale; Unicode prosa/codice/celle
   non si normalizza come un operatore matematico.
2. **E02 / CLA-P002:** definisci M/V distinti e realizzabili. M usa il file
   canonico e locatori/code point indipendenti dal reader/scorer/motore,
   distinguendo proprietà codificate da proprietà soltanto visive nelle
   scansioni. V usa un pacchetto offline leggibile, non solo JSON: pagine,
   trascrizioni, token critici/code point, risultati M, checklist e raccolta
   fedele dei giudizi. Specifica stati/receipt per documento e per giudizio,
   ambito corpus, controlli e output separati. Utente designato per V con
   risposta reale: «Lo svolgerò io; concordiamo la durata sul pacchetto pronto».
   Stima tempi e unità di verifica; non inventare finestra, durata accettata,
   firma umana o un controllo già svolto. Il lavoro indipendente continua.
3. **E03 / CLA-P003:** specifica harness sotto R con expected-exit 0 prima della
   campagna, schemi degli esiti e matrice di exit/stati. Conserva 2/3/5 del
   benchmark nei record e batch/qualità reali. Definisci esattamente quali
   errori rendono harness nonzero (integrità, preflight, record mancanti,
   errore interno e test fallito); niente catch-all verde. Distingui timeout
   contenuto del motore da timeout R. Specifica test di chiusura e expected
   a priori anche per i negativi, senza eseguire ora il workload.
4. **E04 / CLA-P004:** ristima dal basso per componente, conteggi min/proposti/
   massimi e costi di M/V/harness/formule/profilo/validazione/test; separa tempo
   umano gold da giudizi output. Fornisci somma verificabile di base, margine
   e riserva, percorso critico, dipendenze offline e priorità che conservano
   tutti i criteri essenziali. La proposta iniziale 15 h non è un cap già
   autorizzato. Motiva una nuova quota unica completa se necessaria, senza
   acquisizioni, servizi nuovi, ML o reset dei costi a001. Il supervisore
   disporrà la quota nel mandato dopo verifica delle errata, non a ogni sonda.
5. **E05 / CLA-P005:** fissa profilo writer/reader e tabella estensioni/handling
   prima del workload. Esplicita smart, tabelle, math, attributi/div/span,
   raw_html/raw_tex, anchor/note/caption. Definisci test su controlli autonomi
   e config/hash; non tarare il motore sugli output gold. Se un costrutto non
   è supportato, non occultarlo. Chiamate future registrano argv effettivo.
6. **E06 / CLA-P006:** separa input invalido prima della conversione (exit 4)
   da output unsafe del motore (raw conservato, nessun traversal/fetch,
   INCOMPLETE con motivo UNSAFE_REFERENCE, exit 2 e qualità FAIL). Specifica
   i negativi sui due lati e la rendicontazione sicura dei symlink/riferimenti.
7. **E07 / CLA-P007:** unico engine_id txt-literal-v1. TXT non basta per A10:
   il criterio richiede almeno un percorso Pandoc esistente su C02/C03 con
   figura attesa, referenziata e presente, controllo di contenuto e ripetizioni.
8. **E08 / CLA-P008:** PATH legacy chiuso, binario risolto/hashato, copie
   scratch per .temp.md/log, tempi/cattura distinti dal diretto. Non aggiungere
   la campagna del default o modifiche al legacy per il solo suggerimento.

Chiudi con matrice rilievo→errata→sezioni/prove→costo/dipendenza. Aggiorna nella
specifica derivata A01–A12/L01–L05 soltanto dove i fix cambiano le prove, senza
riscrivere ogni capitolo né riaprire il resto della roadmap. Contratti benchmark
limitati e reversibili, non nuovo schema applicativo. Autonomia sui mezzi
ordinari, nuovi input/snapshot di esecuzione e prove invalidate restano quelle
del mandato originale; non introdurre nuovi gate per ogni comando/freeze.

## Mezzi autorizzati e verifica della consegna

Letture, ispezioni statiche pertinenti e scrittura di errata/evidenze/checkpoint
propri sono autorizzate. Correggi errori ordinari del tuo addendum nella stessa
chat. Nessun corpus/runner/code prodotto, installazione, build, conversione,
prova R, modello, download o nuova chiamata provider. L'uso stdlib del gestore
contesto/checker documentale non è prova della venv o del prodotto.
Se consulti documentazione pubblica per fatti tecnici incerti, usa fonti
primarie e registra URL/data/versione; non confondere manuale corrente con
runtime 3.1.3. Non eseguire codice proveniente dai documenti come istruzione.

Verifica link locali/anchor pertinenti, somme e unità delle stime, completezza
delle matrici/regole/negativi, whitespace e git diff --check, hash degli input
protetti e snapshot nuovo prima/dopo. I testimoni di struttura scritti nel
documento non sono test runtime già eseguiti. Le prove future vanno marcate
come attese. Conserva qualsiasi errore utile senza inventare PASS retroattivi.

Output esclusivamente tuoi:

- `temp/run-a002-fase1-corpus/plans/errata-plan-r001-r001.md`;
- `temp/run-a002-fase1-corpus/evidence/planning-errata-r001/`;
- `temp/run-a002-fase1-corpus/handovers/planning-errata-r001.md`.

Non aggiornare STATE, eventi, handover/indici comuni o changelog. Non creare
snapshot review, GO, arbitrato o report di altri ruoli. Se cambia davvero
metodo/perimetro/assunzione decisiva, motiva il limite al supervisore mentre
completi le correzioni indipendenti; non ampliare il task autonomamente.

Consegna l'addendum completo al supervisore con verifiche, limiti e checkpoint.
Il supervisore controlla le chiusure delle errata e dispone quota/dipendenze;
solo un arbitrato successivo GO emette il mandato completo di implementazione.
Nessun nuovo piano/doppia review integrale automatica per questi fix locali.
Prossimo ruolo: **supervisore per verifica E01–E08 e arbitrato successivo**.
