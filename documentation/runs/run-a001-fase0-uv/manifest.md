# Archivio — run-a001-fase0-uv

Nota17ott2026: riferimenti operativi eliminati o già indisponibili sono
resi testo; esiti storici invariati. [Disponibilità delle prove](../../development/cleanup-2026-10-17.md).

## Disponibilità aggiornata — 2026-10-17

La pulizia autorizzata rimuove archivi operativi, duplicati, build e cache
superati; conserva sintesi/esiti, casi decisivi del debito e dipendenze del
prodotto. Le descrizioni di conservazione nelle sezioni storiche sotto si
riferiscono alla data originaria, non alla disponibilità attuale.

[Inventario selettivo e limiti](../../development/cleanup-2026-10-17.md).

- Obiettivo/fase: toolchain Python uv, packaging e verifiche legacy, fase0.1.
- Base/HEAD alla chiusura tecnica:66ba82200e5def5a4db76f9bafccb0731b506091;
  feature/run-a001-uv. Feature commit `8479f975618034abfcfaeb4c9e47ba925d5e356e`;
  squash in dev `b90c6d4c00aed5ceb9122f6f42544736de7ff35a`, genitore uguale alla base.
  Integrazione manuale verificata il 2026-10-07: 405 file MATCH, alberi identici.
  Working tree e indice puliti alla ricezione; successivo commit documentale
  `80c31da46e5caab990d3596c79c44e524ec1dc95` e push verificati il 2026-10-07.
- Esito: **GO / INTEGRATED**, arbitrato implementazione r003.
- Data/responsabile:2026-10-07, supervisore Codex/OpenAI. Nessun deploy/main/cleanup.
- Oggetto review finale:final-s003 SHA
  6295835b1407ea44f6c49780d38061c0fd5d6389a2ef1e096da9dd260f778962.

## Risultato e limiti

Piano r003 approvato; prima review completa e due giri mirati, due reviewer
indipendenti per giro. Ultime review ChatGPT/Claude GO, blocchi chiusi. Quote
temporali **NON PASS**: storico minimo8321,418589/overrun41,418589, upper ignoto;
nuovo osservato8946,997655 e addebito8947,997655, senza upper dimostrato.
Fine turno circa8966,24/overrun86,24 è inferenza, non misura. Gate di finalize
tardivo e record FAIL ricostruito. Nessun workload prodotto oltre cap rilevato.
Il GO accetta la deviazione amministrativa e non sana i costi.

S/B/I/E22/22 e fedeltà12casi, host/standalone/V1 e Docker sono prove dei propri
input/stage con equivalenze verificate. Vecchia C-fast137+300 non attribuita al
nuovo harness;12mock mirati e2R reali a001/b002 attestano il delta finale.
Baseline62/5/perdite e FAIL storici conservati. Nuova architettura, inferenza,
GPU/Windows/V10/V11 e universalità delle ABI non verificate.

## Materiale conservato

Elenco completo con origine, destinazione e hash sorgente/archivio in
archive-files.json (riferimento storico non disponibile). Copie JSON/TXT/PY sono byte-identiche;
i Markdown cambiano solo i collegamenti locali, con trasformazioni in
link-adaptations (riferimento storico non disponibile). Nessun codice prodotto spostato
qui; gli script di evidenza sono conservati come dati storici.

- Arbitrato finale (riferimento storico non disponibile).
- Review finale ChatGPT (riferimento storico non disponibile).
- Review finale Claude (riferimento storico non disponibile).
- Report-r020 (riferimento storico non disponibile).
- Snapshot finale (riferimento storico non disponibile).
- Comandi manuali (riferimento storico non disponibile).
- [Materiale locale/limiti](notes/materiale-locale-conservato.md).

## Verifica e pulizia

Integrità delle copie, adattamento dei link, delta documentale rispetto al GO e
git diff --check verificati dal supervisore; ricevuta di chiusura in temp.
Nessuna nuova esecuzione delle prove, commit/merge/push o pulizia. Tutti i payload
non selezionati e gli originali rimangono nella run locale. I commit reali sono registrati sopra dopo la verifica
dell'integrazione; nessun rilascio o collaudo fuori dal perimetro è implicito.

## Integrazioni dopo la chiusura tecnica

- Verifica del commit dei registri e del push (riferimento storico non disponibile).
- [Valutazione delle pulizie](notes/cleanup-assessment-2026-10-07.md) e
  lista esatta dei payload duplicati (riferimento storico non disponibile).
  Nessuna cancellazione eseguita; archivio266record nuovamente MATCH.
- [Prossimo incremento a002](../../development/next-run-a002-corpus.md), solo PREPARATION.
- [Pulizia manuale completata e ricevuta](notes/cleanup-completion-2026-10-07.md):
  193 file rimossi e copie conservate MATCH, tre tag superati assenti/r007 presente.
  Receipt reali archiviate separatamente; nessuna rimozione del supervisore.

Questi file sono integrazioni amministrative successive, non sostituiscono i
266 record originali di archive-files.json o gli snapshot di review.
