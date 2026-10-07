# Addendum operativo r001 — recupero baseline preliminare

2026-10-05 Europe/Rome. **ADOTTATO** su istruzione esplicita dell'utente nella
conversazione di supervisione. Questo addendum modifica il coordinamento operativo;
non è una nuova review, un nuovo arbitrato del piano o un GO codice.

## Autorità e precedenza

Il [piano r003](../plans/plan-r003.md), SHA
`462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`, e
[l'arbitrato r003](arbitration-plan-r003.md), SHA
`f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`,
restano immutati. NO_GO antecedenti, FAIL e prove storiche restano conservati.
Il [protocollo aggiornato](../../../../development/run-lifecycle.md#mandati-operativi-e-prove-preliminari)
e ADR0007 registrano la decisione permanente; la scope del pilota ne fissa i limiti.

D1 punti1–2: introdotta l'eccezione esplicita per prove **preliminary** autorizzate
prima del freeze ufficiale, con identità per tentativo e nessun valore di accettazione.
Il prompt29 sostituisce il mandato STATIC_PREPARATION_ONLY del prompt28 soltanto
per il pilota identificato nella scope. Nessun riuso delle vecchie scope s005/s006.
D1 punti3–5 e D2–D5 restano applicabili alle prove ufficiali: snapshot del supervisore,
S dopo lo snapshot, current↔S↔snapshot, ricevute e negativi, invalidazione e nuova label.
Il nuovo contesto di ripresa non è lo stage package-s007; non autorizza prove ufficiali.

## Mandato e arresti

La [scope preliminare del supervisore — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
autorizza preparazione r003 e fino a due tentativi offline su nuove directory,
ciascuno con directories→venv→install→pip-check→inventory in tool separati.
La sequenza si arresta al primo FAIL. Nella stessa chat si può documentare la causa,
correggere un difetto ordinario e avviare il secondo tentativo soltanto se le guardie
lo ammettono. Non riparare/riprendere target parziali, né ripetere senza correzione.
Al primo tentativo completo riuscito terminare i tentativi: nessuna seconda prova inutile.

La politica del monitor è ora definita e accettata **solo per il pilota**, prima
che sia implementata: soltanto il temporaneo pyc con suffisso decimale derivato da
un pyc atteso del pip pinned, mentre il proprio seed è attivo. FileNotFound su
quel file consente al massimo due nuove scansioni complete; senza campione completo
nei limiti si arresta. Ogni evento è registrato, input fissi e verifiche finali
restano strict. Non è un audit di codice futuro né una quota atomica. Estendere
questa politica o altre protezioni richiede nuova disposizione del supervisore.

Budget run500MiB/stop384MiB invariato. Nuovo lavoro preliminare cumulativo64MiB
(preparazione fino16MiB inclusa), oltre alla riserva112MiB per la futura esecuzione
ufficiale e16MiB esterni. Misurare run e delta anche allocati; i target storici
sono già nel totale e non vanno sommati due volte. Due tentativi sono un massimo,
non una promessa di spazio disponibile. Nessun cleanup per rientrare nei limiti.

## Ricezione e accettazione ufficiale

Il [prompt29](../prompts/29-implementation-r001-pilot-baseline-recovery.md) identifica
input/operazioni e consegna. Ricezione con [prompt30](../prompts/30-supervisor-baseline-recovery-pilot-reception-r001.md),
che integra le regole di prompt05 e applica questo addendum prima di qualsiasi freeze.
Un inventario preliminare positivo prova soltanto quel tentativo: non è R/S/V0,
PASS baseline, equivalenza della vecchia venv o GO codice.

Dopo ricezione positiva e input stabili, il supervisore valuta la request s007,
aggiorna lo stato tracciato pertinente e congela gli input ufficiali una volta.
Il nuovo scope ufficiale lega i suoi SHA request/budget/politica. L'implementatore
ripete l'esecuzione ufficiale nel target riservato r003: nessuna promozione del
pilota a prova ufficiale. Dopo recupero ricevuto restano R completa D4/nuovo S,
confronti e tutte le prove del piano; V7/V8 e review codice/arbitrato finale futuri.

## Verifica del miglioramento

Raccogliere passaggi di ruolo, tempi effettivi, byte logici/allocati e tentativi.
Dimostrare negativi su helper/input mutati, scope/stage errati e receipt di altro
tentativo prima dei figli operativi. Review indipendenti e controlli di contenuto
restano richiesti. Non dichiarare il processo più veloce o affidabile prima del pilota.
