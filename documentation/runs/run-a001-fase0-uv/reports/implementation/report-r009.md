# Implementazione r001 — report r009

Mandato43/r011, stage package-s013. **WAITING_FOR_SUPERVISOR_RECEPTION**.
Tre priming nativi transitivi riusciti, cache acquisita139nomi/25aggiunte con
provenienza e sigilli; lock/check non completati per arresto risorse reale.
R/D4 PASS per ciascuno dei tre lock, senza trasferire i PASS precedenti.

Correzione ordinaria: template --no-build + --only-binary incompatibile nella
CLI uv0.10.10. Sessione97486/exit2 prima della risoluzione, workload0s. Log,
receipt e variante conservati. Rimossa soltanto --no-build dal diagnostico;
--only-binary=:all: conserva il divieto globale di sorgenti/backend. Transitive,
--no-sources/config vuota ricevuta/managed3.12.13/keyringdisabled/env19 invariati.
La proposta precedente --no-deps era insufficiente per metadata; r011 la corregge.

Priming r002/session57522 exit0:28package,135nomi cache (21nuovi), circa1.16s.
Lock r001/session74792: R PASS, nativeexit1/wrapperexit2,0.158687s; cache
insufficiente per universal2/arm64, mpmath1.3.0 e hf-xet. Nessun conflitto provato.
Priming r003/session72768 exit0:7package,136nomi, variante universale osservata.
Lock r002/session99702: R PASS, nativeexit1/wrapperexit2,0.146469s; miss hf-xet
abi3/Darwin, transformers4.57.5 e Brotli. Varianti non mutano pin/grafo prodotto.
Priming r004/session89031 exit0: completati quei candidati e139nomi/25aggiunte.

Lock r003/session36068: R PASS, workload0.206362s; monitor stop output_cap,
native/launcherSIGTERM(-15), nessun timeout. Stderr1164160byte supera1048576
di115584byte; simultaneamente registri oltre8388608byte. Trace interrompibile
prima del completamento:171URL metadata mancanti nel parziale, soprattutto
fonttools[woff]/Brotli e backtracking su numerose versioni. Nessun lock prodotto,
check NOT_EXECUTED, nessun audit TOML o conflitto del grafo dimostrato.
Receipt inside.json e runner.json reali; **receipt.json esterna del wrapper
non prodotta** nel terzo tentativo, nessuna receipt inventata.

La guardia periodica0.5s non è una quota atomica. Il burst ha superato stream e
registri fra i campioni; overshoot e parziali preservati. Corretto il monitor
autore per non sovrascrivere il primo motivo di arresto con output_cap: diff/hash
in monitor-correction-r002.json, receipt storiche immutate. Nessun nuovo workload
dopo il cap. Raccolti tutti i PID osservati propri/starttime; nessuno vivo.
Massimo gap osservato 0.500104s; nessun picco istantaneo dichiarato.

Ledger run+.venv-python lstat/no-follow e Hentry525762560 invariati. Misura alla
stesura: H=546529280, Delta=20766720, riserva residua32MiB=12787712.
Gate globale32MiB/stop896MiB/libero1GiB PASS, cap registri8MiB **FAIL**:
9715712byte alla stesura; cap log1MiB FAIL. Nessun ammorbidimento o nuovo budget.
Nuovi inventari/sigilli JSON compressi per evitare copie inutili; precedenti
intatti. Input congelati request/scope/policy/config classificati nell'attività
metadata_and_inputs; registri/strumenti/output nel cap registri. Ledger totale
non cambia; anche il conteggio conservativo iniziale è oltre cap a fine tranche.
La raccolta finale aggiunge soltanto prove e consegna dopo l'arresto, non workload.

Audit cache in cache-acquisition-audit.json.gz: inventario pieno sigillato,
delta rispetto759record storici,139nomi esatti/25nuovi e provenienza trace/seed.
Ogni lock ha cache identica pre/post; source/build/Git solo marker tecnici.
URL loggati online solo PyPI/files.pythonhosted.org; catene redirect/header e
body/wire non esposti dal frontend: NOT_MEASURED, nessuna whitelist atomica o
claim solo-sidecar. Nessun archivio wheel/sdist completo lasciato nel delta
cache; ciò non esclude una lettura streamed temporanea. Nessun hook/backend,
sdist/Git/install/pesi/font o payload ML completo osservato.

Startup bootstrap/managed/baseline enumerato pre-probe e dopo, origine/versioni
e binari/BUILD/constraints/coverage/config ricevuti verificati. Integrità baseline
1892file/211directory PASS_PRESERVATION_ONLY. S013 MATCH pre/post. Root/copia16
input invariati, metadata Ebook0.18/lxml/six riconosciuti nelle trace; senza lock
non attestare sdist hash nel lock o copertura universale/ABI/installabilità.

Seguito pronto NOT_AUTHORIZED in next-gate-request-r001.json: priming diagnostico
universale fonttools[woff]4.66.1/Brotli sui miss reali, sigillo e lock/check R/D4,
senza nuovo planning-only. Necessaria ricezione dei caps esauriti: proposta
registri12MiB/log2MiB mantenendo totale32MiB e Hentry invariati, oppure ricezione
di log native senza debug verbose, preservando tutti gli esiti/log vecchi.
Nessuna variante eseguita con questi caps. Metadata stimati0.5MiB, registri fino
2MiB; budget e stdout nativi da ammettere prima. Deadline cumulative restano
900/900/120 meno le durate reali. Input diagnostici mai promossi a prodotto.

Baseline caratterizzazione PASS/suite62pass5fail e perdite; s009 byteFAIL e
lacune startup/rawhook invariati. Product S/B/I/E/V pertinenti aperti, V7/V8
pesanti distinti, V10/V11 esclusi. Review reali ChatGPT/Claude e arbitrato finale
necessari. Nessun GO finale, Git manuale, niente commit/merge/push/deploy/cleanup.
Report-r008/delivery/checkpoint d'ingresso preservati. Trasferire file reali
temp/managed/cache/work e lavoro non committato. Prossimo supervisore05+r011.
