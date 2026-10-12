# Checkpoint implementatore a002 r009

Consegna completa 2026-10-10T15:07:18.445287+00:00; provider OpenAI/Codex basato su GPT-6;
deployment/effort/chat ID non esposti. Nessuna delega/review/GO autore.
Branch feature/run-a002-corpus, HEAD/dev/origin-dev/mergebase
80c31da46e5caab990d3596c79c44e524ec1dc95, indice vuoto, Git manuale dopo GO.

[Report finale](../reports/implementation-r009.md) contiene nota iniziale,
delta, matrice, prove/fallimenti/gate/costi e limiti.
[Report permanente](../../../../benchmarks/2026-10-10-a002-r009-confini-reader.md),
[dati](../../../../benchmarks/2026-10-10-a002-r009-confini-reader-data.json).

Reader observations-v9-reader-boundaries SHA256
 a8ffa9eb44cf1695ca2e7e611c91ef8c7594605fe38a3bb194811f056cdbc737.
F39 guardia completa prima della proiezione; F40 tokenizer condiviso;
F41 dominio/mappa producer; F42 interazioni finite e consegna documentata.
4 file esistenti modificati,3 nuovi; ricevuti77011artefatti invariati.

Prove reali: matrice308/308 (v8 identico178/308);339test r009;
1712mirati PASS/16F04deselected;1869fast PASS/310subtest, finale stage-s006;
35fixtureCLI/70exitOS attesi;H01–H08 PASS,H12±FAIL voluti;
64raw rescore0delta numerico/gate,4PASS/28FAIL per set32;checker indipendente
aritmetico/catena/raw, non allineamento completamente indipendente.
GoldPROVISIONAL/nuovaV0/nuovi motori0.56AVAILABLE/EXACT_GLOBAL_DISTANCE,
8UNAVAILABLE/TRIVIAL_ZERO, non fullPASS. L01/L02NOT_MEASURED,L04UNAVAILABLE,
L05lower/sampled/hardware non certificati,ChromeNONPASS/gap storici preservati.

Ultimo stage effettivo evidence/implementation-r009/stage-s006: S/B/I,
wheel canonica reinstallata,dieci hash e managed3.12.13
identico nei figli. input-bytes-*-before/after e input-store conservano ogni
versione consumata. [Manifest consegna](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r009/final-delivery-inputs-001.json).
Snapshot esecuzione implementation-r009-delivery-001, SHA2561e54271bdf8dfc0a2f1ed6a7212b57c4ee2427c2ace5ce6c486dbb67f7c11892,
worktree61d97379a7e64cdd99628d57803b21f2a3ef87f702936b948dcf2ccce872f73d, 328file/5artefatti,
verify MATCH; output live esclusi, non comune né approvazione. Snapshot r008
ricevuti immutati, S007 solo storico. Il supervisore includerà anche report e
checkpoint finali nel nuovo comune.

Risorse a2026-10-10T15:05:03.178360+00:00: wall201488.809292s;
CPU accounting12793.622141s lower, propria
963.321937s;45s storici approssimati separati;
retained noto2696144109B;RSSsampled749760512B.
Ledger e dettagli in evidence/implementation-r009/budget-delivery-001.json,
ledger.jsonl e process-and-resources-final-001.json. Nessun reset, nessun
workload/processo pendente; reaping/socket verificati, rete/invii/servizi0/€0.
Stop nuovi workload2026-10-11T06:37:31Z,deadline09:37:31Z,riserva3h;
CPUstop19800/discostop5GiB/RAMsampledstop6GiB. Amministrazione non interamente
misurata; il wrapper doc-check colliso ha gapCPU esplicito, non sanato.

Fallimenti operativi sono conservati e corretti (EPERM socket con escalation
R isolata; fixture paragraph prima292; CLI001; label/parent H01; percorso
protocollo nel delivery checker/link; collisione receipt doc-check).
Nessun rifiuto effettivo auto-review, nessun confine/isolamento indebolito.

PROSSIMO RUOLO: supervisore per ricezione, nuovo comune e due NUOVE review
indipendenti ChatGPT/Claude, poi arbitrato; aggiornamenti condivisi suoi.
Nessun GO autore, Git/fase2/cleanup automatici. Non riaprire il piano per i
fix locali già conclusi e non trasferire PASS storici alle nuove versioni.
