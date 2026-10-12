# Arbitrato implementazione a002 r007 — NO_GO locale

2026-10-10 Europe/Rome. Supervisore Codex/OpenAI, identificativo esatto del
deployment/chat non esposto. Applicate le skill manage-implementation-run e
verify-conversion-fidelity. Letti integralmente entrambi i report/checkpoint
r007 e le tracce, poi scanner e consumer pertinenti. Review reali di nuove
chat indipendenti ChatGPT/Codex OpenAI e Claude/Anthropic, provider/modello
come dichiarati nei report. Nessuna review simulata, sostituzione o delega.

**NO_GO** sull'oggetto `implementation-r007`: tre cause bloccanti in quattro
rilievi. F35–F37 sono correzioni locali; F38 comprende documenti e prove.
Nessun nuovo piano. Il [prompt27](../prompts/27-implementation-r008-fix-scanner-progress-and-inline-domains.md)
è preparato per la stessa chat implementatrice. **Esecuzione oltre lo stop
non disposta**: occorre la variazione temporale esplicita descritta nella
[proposta risorse](proposed-resources-r008.md). La decisione tecnica NO_GO
è distinta da questa dipendenza operativa.

## Oggetto e validità

[Comune r007](../../../../../temp/run-a002-fase1-corpus/snapshots/implementation-r007.json): SHA256
`6a5235582225134b364818c96d0e3029597d505d39db936b16e0bec039de7ab4`,
impronta `296a3cc7b90044cb6abb2f38fcf2c5d030820e34f03d361a45d078c5c329b535`,
322 file generali/65669 artefatti espliciti, inclusi 20 PDF versionabili.
Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base
`80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto, nessuna query remota.
MATCH prima/dopo per entrambi i reviewer e alla ricezione supervisore prima
dei nuovi registri. Le review valgono per il medesimo oggetto.

- [ChatGPT](../reviews/review-implementation-r007-chatgpt.md): NO_GO,
  GPT-R007-I001, traccia statica label/code soft-line. Ingresso 08:14:41Z.
- [Claude](../reviews/review-implementation-r007-claude.md): NO_GO,
  CLA-R007-I001/I002/I003, tracce statiche T1–T3. Ingresso 08:14:48Z.
- Entrambi dopo lo stop 08:06:54.369068Z: **nessuna sonda runtime propria,
  nessun nuovo R/test/CLI/snapshot tecnico reviewer**. MATCH/hash/JSON e
  analisi statica sono distinti da prove di esecuzione. Le conseguenze su
  COMPLETE, blocchi/asset e non terminazione sono dedotte dai rami, non
  exit OS o aperture di file misurati in questo giro.
- Entrambi controllano le 64 misure ricevute: 4 PASS/28 FAIL per set32,
  catena coerente e numeri invariati. Il checker ChatGPT ricalcola le
  distanze dei flussi registrati, non l'allineamento dei blocchi. Le suite,
  38 casi CLI/76 exit e 24 R (22 PASS/2 FAIL H12 deliberati) sono prove
  dell'autore ricevute, non rieseguite da reviewer o supervisore.

La [ricezione](../../../../../temp/run-a002-fase1-corpus/evidence/supervisor-implementation-arbitration-r007/reception.json)
lega report, checkpoint, evidenze e tracce con hash. Le premesse statiche
sono in [static-premises.json](../../../../../temp/run-a002-fase1-corpus/evidence/supervisor-implementation-arbitration-r007/static-premises.json).
Il supervisore non importa né esegue reader/audit/scorer/R/motori; lo stdlib
manager serve a identificare il contesto. Nessun input prodotto modificato.

## Decisione su tutti i rilievi

| Rilievo | Decisione/attribuzione | Evidenza, raggiungibilità e impatto | Correzione/prova di chiusura |
| --- | --- | --- | --- |
| CLA-R007-I001 | Accolto, alta, **bloccante**, **F35**. Regressione del ramo raw LaTeX introdotto da r007. | `observations.py:147–163,178–191`: fence `{=latex}` imposta limit prima della blank; sotto raw_fence il limite non si aggiorna. Backslash/backtick dopo blank ottiene bound<i: `i=min(i+2,bound)` o `i=bound` torna alla blank, con stato identico. T1 dimostra un ciclo raggiungibile da observe/audit/measure. Fence dichiarato ammesso, nessuno nei 64 raw/test ricevuti. Il reader r006 saltava il fence e terminava. | Avanzamento garantito e dominio raw LaTeX coerente con il consumer, comprese blank/marker del corpo e chiusura/EOF. Sonda prima con timeout breve e reap, poi terminazione dopo; raw/offset/diagnosi e blocchi esterni preservati. Un watchdog senza fix non chiude il difetto. |
| CLA-R007-I002 | Accolto, media, **bloccante**, **F36**. Residuo F31/F29; r007 chiude il caso tra blocchi, non nello stesso dominio. | `observations.py:189–198,224–227` salta al bound su code/math/tag non chiusi; un opener vero `<!--`/CDATA prima di quel bound non viene visitato. T2: stessa riga/soft-line, poi commento oltre blank con asset e paragrafo gold. Il consumer usa leaf_end, ma lo scanner non produce la barriera. Asset/prosa diventano attivi e missing può essere mascherato; audit COMPLETE con asset presente è dedotto, automatic gate non PASS per unknown. Non è una prova di rete o evasione. | Delimitatore incompleto non disabilita il riconoscimento di token realmente aperti nel medesimo blocco. Commento/CDATA intero fino a chiusura/EOF, zero prosa/formula/ref/aperture dal corpo, omissioni reali visibili. Positivi letterali chiusi nel dominio, prosa esterna e chiusure in blocchi diversi restano discriminanti. |
| CLA-R007-I003 | Accolto, media, **bloccante**, **F37**, accorpato al rilievo ChatGPT seguente. Regressione r007 della label. | `observations.py:183–191,348–366`: markdown_label accetta code chiuso sulla soft-line, scanner tronca il bound al newline. `<!--` dentro code diventa falso span EOF; unità e blocchi successivi assorbiti, inline poi legge il code senza unknown. T3, input ammesso e senza asset, audit COMPLETE senza diagnosi dedotto. | Stesso intervallo code fra walk/scanner/inline entro il producer finito. Conservare input valido: tre blocchi, tipi/ordine/testo/ref/raw/offset corretti, zero token spurio e COMPLETE. Varianti CDATA/due backtick/trasporti pertinenti, code ordinario e veri token inerti. |
| GPT-R007-I001 | Accolto, media, **bloccante**, stessa causa **F37**, nessun doppio fix. Regressione r007. | [Traccia indipendente](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/static-label-trace.md) concordante con T3: `See [Known ` + code su due righe + link, heading e paragrafo. Lo scanner applica un limite che il walk/consumer non applicano. Heading presente diventa perdita inventata del misuratore. | Criterio F37 comune; il rifiuto generico dell'input valido o unknown globale non risolve la divergenza. Non espandere le chiusure ai blocchi estranei per compensarla. |

La traccia di F35 è sufficiente alla decisione statica: il limite resta
inferiore al cursore e i rami possono ripristinare uno stato già visitato.
F36/F37 descrivono divergente classificazione dei medesimi byte, non nuovi
formati da aggiungere. Nessun PASS falso viene inventato, né la presenza
di INCOMPLETE rende accettabili reference/prosa/metriche spurie. L'assenza
degli input nei 64 raw pubblicati non chiude il contratto del misuratore.

## Chiusure e prove da preservare

F32 autolink chiuso nel dominio revisionato; F27/F30 e F28 locale chiusi,
F26 e F16 sui64 preservati. F31 parziale via F36, F33 parziale via F35/F37;
F34 richiede aggiornamento via F38. Residui globali F29/F09/F13/F14 restano
circoscritti alle cause accolte, nessuna riapertura indiscriminata.
Corpus/gold/raw/V13/A03 invariati; L04 formule output UNAVAILABLE distinta
da V riferimenti. L01/L02/L05, Chrome NONPASS e helper12 gap, errori checker,
test rossi intermedi, E9 con atteso errato e preflight STALE conservati.

F38: protocollo/due guide e nuovo report/dati permanenti descrivono solo
ciò che le nuove prove chiudono; report r007 e tracce restano storici.
Test mirati/regressioni, suite veloce dopo preflight S/B/I/R corrente,
CLI dipendenti con exit OS e rescore/invarianza64 sono dovuti ai byte nuovi,
senza conversioni motori o V umana nuova. Nessuna campagna a001/API/ML/Docker
o scelta dei motori. Implementatore identifica input e snapshot tecnici
autonomamente; due NUOVE chat reviewer dopo consegna completa r008.

## Tempo e prossimo ruolo

Quota vigente 52h dall'entry `2026-10-08T07:06:54.369068Z`, attese incluse.
Stop nuovi workload `2026-10-10T08:06:54.369068Z`; deadline amministrativa
`2026-10-10T11:06:54.369068Z`. Nessun reset né estensione eseguita qui.
Il ledger autore ~40,625h/10400,457453 CPU-s lower/2086483026 B retained
è storico; la receipt aggiorna wall reale, non certifica saldo CPU/disco.
Costi reviewer e amministrativi da integrare senza doppio conteggio.

Il [contesto tecnico fix-r007](../../../../../temp/run-a002-fase1-corpus/snapshots/implementation-fix-context-r007.json)
conserva gli input r007 e aggiunge review/arbitrato dopo soli quattro registri
generali. Non è GO, non è disposizione di risorse e non riscrive il comune.
Prossimo ruolo: supervisore per disposizione temporale effettiva, poi
stessa chat implementatrice con prompt27, consegna completa, nuovo comune,
due review indipendenti e arbitrato reale. Git rimane manuale dopo GO finale.
