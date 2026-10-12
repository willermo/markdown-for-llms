# Implementazione a002 r008 — avanzamento scanner e domini inline

Agisci esclusivamente come **implementatore**, nella stessa chat autrice di
r007. Completa F35–F38 e fix ordinari nella stessa chat, con prove dipendenti
e consegna completa r008. Nessun nuovo piano. **Mandato operativo con
disposizione temporale24h effettiva**: [addendum risorse r008 r001](../arbitrations/addendum-resources-r008-r001.md)
autorizzato dalla risposta utente. Puoi procedere nel perimetro F35–F38
e con i fix ordinari; la proposta12h/PENDING precedente resta storia.

## Ingresso, identità e letture

Run `run-a002-fase1-corpus`; branch `feature/run-a002-corpus`, HEAD/dev/
origin-dev locale/merge-base `80c31da46e5caab990d3596c79c44e524ec1dc95`,
indice vuoto alla ricezione. Commit/merge/push manuali utente dopo GO finale.
Piano r001+errata e GO piano, mandati05/06/09/12/15/18/21/24 e addendum V
restano validi nei requisiti pertinenti. Prevale
`arbitrations/arbitration-implementation-r007.md`, **NO_GO**: F35–F37
bloccanti e F38 completamento documentale/probatorio. Sono tre cause in
quattro rilievi, non quattro implementazioni indipendenti.

Percorsi della run relativi a `temp/run-a002-fase1-corpus/`. Leggi nell'ordine:

1. AGENTS, temp/HANDOVER, parte iniziale PROJECT-CONTEXT, STATE/HANDOVER/brief
   e `handovers/supervisor-resources-r008-r001.md`; checkpoint arbitrato r007 storico.
2. Le skill manage-implementation-run e verify-conversion-fidelity, protocollo
   run-lifecycle, questo prompt, arbitrato r007 integrale e addendum risorse
   r008 r001 effettivo. La proposta12h PENDING non è la disposizione corrente.
3. **Entrambi** report/checkpoint r007 ChatGPT/Claude, relative tracce statiche
   `static-label-trace.md` e `static-traces.md`, audit e verify. Nessuna sonda
   runtime reviewer r007; gli effetti sono deduzioni statiche da riprodurre.
4. Consegna completa e chiusura finale report/checkpoint autore r007, prove
   finali attribuite e report/dati permanenti `2026-10-10-a002-r007-contesti-markdown*`.
5. Reader integrale: markdown_lexemes, scope, code/math/tag, label/link,
   markdown_units, inline/barriere, fence/producer/pipe/metadata/attrs e
   trasporti. audit/observe_bundle/scorer/safety pertinenti, test r003–r007.
   ADR0001, piano/errata/workflow e guida uv per requisiti/toolchain. Niente
   scansione indiscriminata a001 o rilancio di vecchie campagne.

Comune revisionato `snapshots/implementation-r007.json`, SHA256
`6a5235582225134b364818c96d0e3029597d505d39db936b16e0bec039de7ab4`,
impronta `296a3cc7b90044cb6abb2f38fcf2c5d030820e34f03d361a45d078c5c329b535`,
322 file/65669 artefatti. MATCH nelle due review e alla ricezione prima dei
registri; diventa storico dopo i quattro documenti supervisore. Non rigenerarlo.

Ingresso tecnico corrente `snapshots/implementation-fix-context-r007-resources-r001.json`:
SHA256 `c43903ff8f3b4155fc643c3ff7e4118f964b8f40d8849a518a75de3f0d6b6889`, impronta `bdb4142a91a4ca7d7b15d454ee4caebbd4da3d264fdbf0359498784da3efea53`. Conteggi/identità/transizione
in `evidence/supervisor-resources-r008-r001/snapshot-receipt.json`.
Il contesto fix-r007 precedente e il comune rimangono storici. Solo quattro
registri generali aggiornati per la disposizione temporale, prodotto/test/
corpus e prove ricevute invariati. Manager stdlib, non runtime o GO prodotto:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-fix-context-r007-resources-r001
```

Registra argv/cwd/output/exit e Git/delta anche non committato, hash dell'addendum
e fonte utente. La variazione temporale è delta amministrativo esplicito,
nessun restauro di input per mascherare STALE degli snapshot storici.
Dopo modifiche autorizzate prepara autonomamente manifest/gate/snapshot di
esecuzione r008 con label nuove, output vivi esclusi. Nessun freeze intermedio.

## Contratto e correzioni complete

Scanner dei confini e consumer devono riconoscere gli stessi intervalli e
domini finiti. Ogni transizione dello scanner deve avanzare o consumare uno
stato finito dimostrabilmente terminante; nessun bound inferiore al cursore
può far tornare su stati già visitati. Un vero token inerte fuori dai letterali
chiusi resta intero sino a chiusura/EOF, anche attraverso blank/ATX/fence/list/
quote/nota. Un opener dentro code/attributo/comando già consumato è letterale.
Un delimitatore incompleto non sopprime lo scanning dei veri token nel medesimo
blocco. I blocchi esterni, tipi/ordine/testo/ref/asset/raw/offset restano corretti.

Scegli mezzi reversibili entro il contratto, anche riorganizzando la logica
finita. Non basta una guardia timeout o un assert sui soli payload, né un
rifiuto generico di input già ammessi. Non estendere lookahead a blocchi
estranei, HTML/Markdown/TeX universali o adattare gold/scorer al reader.

### F35 — terminazione e dominio del fence raw LaTeX

CLA-R007-I001/T1. `{=latex}` con `\begin{center}`, blank e `\end{center}`:
limit rimane prima della blank, backslash/backtick post-blank porta i indietro.
Vale per observe, audit, percorso di misura. Registra il prima su copia r007
con **timeout breve** predeterminato, reap del gruppo e receipt distinta dal
normale R PASS. Non spendere900s per confermare un loop; nessun processo orfano.

Correggi progressione e dominio del raw fence nel suo consumer. Verifica
blank+backslash/backtick, marker del corpo che oggi interrompono scope,
chiusura effettiva/EOF e blocchi dopo la chiusura. Positivi raw LaTeX finito,
fence letterali e raw HTML discriminano. Termine, raw/offset/diagnosi e
contenuti esterni devono essere provati; non rendere supportata tutta la TeX.
Unknown raw/semantiche già dichiarate conservate e localizzate.

### F36 — vero commento/CDATA dopo delimitatore incompleto nello stesso blocco

CLA-R007-I002/T2. Prefissi backtick spaiato, `$x`, `$$`, `<alpha` seguiti da
`<!--` o `<![CDATA[` **nella medesima riga e sulla soft-line dello stesso
blocco**, poi blank/heading/immagine/paragrafo dentro il token. I test r007
con delimitatore in un blocco separato non coprono questa interazione.

Token intero a chiusura/EOF, prosa/formule/ref/asset del corpo assenti dalle
foglie attive, zero aperture/hash/UNSAFE_REFERENCE per asset commentati anche
`../`, blocco/page missing davvero omesso non soddisfatto dal commento.
Unsupported/diagnosi nel profilo html-tokens-v1 visibili, audit INCOMPLETE
e exit2 coerenti, unknown non wildcard. Positivi code/math/tag davvero chiusi
nel dominio, `<` prosa/valuta, chiusure nei blocchi diversi e veri asset attivi
restano discriminanti. Non convertire la deduzione COMPLETE del reviewer in
una receipt runtime preesistente; fissa attesi prima e misura realmente.

### F37 — code soft-line nelle label

GPT-R007-I001 e CLA-R007-I003/T3, stessa causa. Caso minimo valido:

````markdown
See [Known `first
<!--` words.](https://example.invalid)

# Visible heading

After words.
````

Atteso: paragraph `See Known first <!-- words.`, heading `Visible heading`,
paragraph `After words.` in ordine; reference `https://example.invalid`,
COMPLETE, zero token spurio. Il walk già accetta il code soft-line, scanner
tronca al newline. Uniforma l'intervallo nel producer finito senza chiusure
prese da altri blocchi. Copri link e image-label pertinenti, due backtick,
CDATA/PI/declaration nel code, quote/list/nota nei trasporti effettivi e EOF
label reali. Positivi code su singola linea e soft-line fuori label; vero
token fuori code deve ancora rimanere inerte intero. Assert su struttura,
testo/raw/offset Unicode/reference/diagnosi: un COMPLETE solo non basta.

### F38 — documenti e attribuzione delle prove

Rettifica PROTOCOL e due guide con contratto effettivamente verificato.
Nuovi report/dati permanenti datati alla consegna, storia r007 immutata.
Documenta causa/minima scelta e risultati reali prima/dopo, nuovi readerID
e input; nessun PASS retroattivo. Distingui sonde/gold PROVISIONAL/stubCLI
da baseline dei motori/V reale; condizionali/bound globali da metriche complete.

## Prove, toolchain e consegna

Prima del primo managed: origine locale CPython3.12.13/startup/package e
archivi/installazione dei dieci moduli effettivi. Dopo delta S/B/I,
sdist/wheel canonica ricostruita e reinstallata, dieci hash e preflight
attuale prima di --no-sync; package installato nello stesso interprete dei
figli. Vecchi S006 generali STALE per documenti supervisore non diventano
PASS. Equivalenze limitate ai byte realmente consumati con copie/hash/
manifest prima-dopo, non equivalenza per nome o controllo versione.

R/Firejail senza rete, stesso interprete dei figli, shell=False, registry/
PATH/safety/resource caps invariati. I nuovi input runtime sono autonomi;
non inventare gating su output ancora da produrre. Prima/dopo sul reader
r007 conservato dove utile, input/attesi fissati, errori tool preservati e
corretti nella stessa chat. La prova del loop è piccola e sorvegliata.

- Matrice finita F35–F37 con tutte le interazioni nominate, positivi e
  negativi sul percorso audit→observe→score. Assert su contenuto/stati/
  diagnostica/localizzazione/denominatori/asset effettivamente aperti,
  completezza/qualità separate. Gold delle sonde provvisori, zero firme V.
- Regressioni mirate dei domini r002–r007 dipendenti dal reader; nuova suite
  veloce dopo preflight/R corrente: `"$RUN_DEV_PY" -I -B -m pytest tests/
  --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker -q`.
  Smoke radice esclusi; non suite universale o campagna a001.
- CLI run/evaluate su casi discriminanti con exit OS reali, run.json/final/
  errori tracciabili, writer sintetico dichiarato. PROVISIONAL permette
  automatic gate PASS con qualità INDETERMINATE/evaluate5 atteso. Non
  scegliere expectedexit dopo il risultato. Contratti H dipendenti e H12
  FAIL deliberati conservati; ripetere quelli invalidati dal delta reale.
- Nuovo rescore o prova di invarianza sui64 raw con input finali, ID/versioni/
  catene attuali e checker aritmetico indipendente. Due set32,4PASS28FAIL
  da verificare, rawtree e conversion_final/run/old_* storici distinti da
  final attuale. **Zero motori/V nuova** per il delta previsto; un mutamento
  numerico va spiegato e pubblicato, non corretto sul gold per forzare invarianza.

F32/autolink e F27/F30/F28locale/F26/F16 preservati; F31/F33/F34 chiudono
tramite nuove prove pertinenti, residui globali non chiusi per solo conteggio.
Scorer non si modifica per compensare il reader; bound globale non è bound
del CER/WER strutturale, TRIVIAL_ZERO non misura completa0. Corpus/gold/raw/
V13/A03 e L04output UNAVAILABLE distinta invariati; L01/L02/L05/ChromeNONPASS/
helper12gap/fallimenti storici conservati. Nessun nuovo motore/browser/render,
M/V generale, API/Docker/ML/pesi/provider/download/macchine/fase2 o Git automatico.

Output tuoi: `reports/implementation-r008.md`, `evidence/implementation-r008/`,
`work/implementation-r008/`, `handovers/implementation-r008.md`, nuovi report/
dati permanenti. Matrice fix/requisito/prova/esito/costi/limiti e input finali;
nessun processo pendente, nessun GO autore. Non modificare registri comuni,
arbitrati, reviewer o checkpoint supervisore. Alla consegna completa:
supervisore, nuovo comune, due NUOVE review indipendenti mirate, arbitrato reale.

## Risorse e recupero

**DISPOSED24h** con addendum risorse r008 r001 e risposta utente conservata.
Ancoraggio al primo clock dopo ricezione2026-10-10T09:37:31Z (timestamp
piattaforma non disponibile), non al tuo avvio: tutto incluso, anche
preparazione supervisore, attese e review. Stop nuovi workload2026-10-11T06:37:31Z
(08:37:31 Europe/Rome); deadline2026-10-11T09:37:31Z (11:37:31), riserva3h.
Entry originale2026-10-08T07:06:54.369068Z conservata: nuovo limite wall
cumulativo268236.630932s, stop257436.630932s.
Vecchia quota52h e stop restano storia, nessuna sanatoria/reset/retrodatazione.
Costi previsti servizi€0/rete0. Autorizzazione per risultato e fix, senza
un nuovo consenso per ogni comando ordinario. Stop al primo limite tempo/CPU/disco.

Limiti cumulativi invariati: CPU21600s/disco nuovi6GiB/RAM8GiB/stop sampled6GiB,
stopCPU19800s/disco5GiB e riserva1800CPU-s/1GiB; R≤900s/comando,1MiB/stream,
32MiB/file. Ledger autore storico146250,572159wall-s/10400,457453CPU-s lower/
2086483026B retained/RSSsampled740855808B non è saldo attuale. Integra costi
reviewer/amministrativi senza doppio conteggio, aggiorna wall incluse attese
e misure CPU/disco/RAM; ignoti restano tali. Nessun upper/hardware certificato.

Diagnostica/correggi/riprova errori ordinari,
launcher/argv/path/timeout/permessi nella stessa chat, preservando fallimenti
e input di confronto. EPERM/EACCES non è rifiuto approvazione: escalation tool
per azione già autorizzata mantenendo isolamento; rispettare rifiuto effettivo.
Non aumentare quote o ridurre gate. Solo scostamenti sostanziali/limiti reali
tornano al supervisore; snapshot di esecuzione non sono nuove approvazioni.
