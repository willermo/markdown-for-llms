# Implementazione r001 — report r018

Esito: **BLOCKED_CORE_TIME**, implementazione non conclusa; nessun GO finale.

Mandato55: sei rilievi arbitrati, senza modifiche degli algoritmi. Stato dettagliato in `evidence/implementation-r001/fix-review-r001/matrix-final.json`; comandi/exit/costi e FAIL reali nella delivery.

## Realizzato e verificato

- Modalità standalone esplicita (`--test-mode` / `RUN_TEST_MODE`), default official e conflitti rifiutati; run_id/schema/path confinati e verificati da run_context. Git standalone supporta clone locale senza commit. Le prove reali standalone e run_id diverso restano da eseguire.
- Compose entrambi corretti: network none, contesto pubblico preparato, due additional_contexts verificati, pin OCI/APT. Tool permanenti `prepare_docker_inputs.py` e `make_input_inventory.py`, senza dipendenza da helper temp o HOME personale. Supply locale105 payload/29deb, hash/RECORD/ABI/firme/resolver nativo, senza download. Build reale Compose r006 e r007 PASS; i due config CPU build sono equivalenti, GPU solo config. R007 `sha256:b3db6dc22d92d7771aba40013bab4281ebe9f68188d5c46fc2617136b7d2c3a8`. Il contesto consumato è byte-equivalente agli input finali59; equivalenza puntuale allegata.
- C-docker versionato: pytest `tests/docker/test_marker_contract.py::test_marker_contract` PASS, nove sottocasi offline; startup a container fermo/full I prima degli import, netnone/nomount/pullnever/capdrop/nopriv/nohealth/mem-pids e raccolta propri container. Il FAIL IndentationError è conservato; prefisso stdin corretto e AST validato prima Python.
- README/guide durevoli, AGENTS corretto, JSON/backend/multi-formato/polling/scenari riconciliati col codice.74 blocchi+9inline individuali con identità originali/destinazioni/motivi; link e whitespace PASS. Il README finale precede B56. Hash originali normalizzati r003 sono mantenuti separati dagli hash effettivi delle fence del README originale.
- Socket cleanup sul path reale `s`, errore primario preservato/cleanup secondario distinto/receipt sempre.3 mirati PASS_MOCK_ONLY e nuovi R reali PASS; nessun indebolimento R/D4.
- B56 uv nativa offline sdist→wheel dalla stessa sdist, attestazione84 prima/hook reali raw, audit archivi/metadata/RECORD PASS, canonica installata in sei prefix e seed probe prima R. S/B build inputs esatti rispetto S59: dieci moduli/pyproject/lock invariati; README canonico corrente.
- Sei I/E58 PASS. Dopo correzione della fixture mock Git: S59 ufficiale e I/E59 root/base-a/base-b/esterno PASS. Dev59 FAIL al limite; API59 non tentata. Non attribuire I58 al target finale59.

## Errori e limiti

Supply tentativi1–3: alias ABI/nome wheel, RECORD vendored, forkCPU/CUDA; conservati e corretti, quarto reuse-in-place PASS. Driver Compose: RepoDigests/path GPU corretti in versioni esclusive. La contabilità conservativa cache+immagini duplicava cache Shared: il monitor aggiornato usa il dato nativo Shared, immagini proprie intere e floor storico11247782439, senza cleanup/reset; due cache-only review comprese.

Fast56 ha eseguito137 test e300subtests PASS ma E/outer FAIL per snapshot invalidato durante diagnosi: non è PASS ufficiale. Fast57 è stato fermato ledger_cap. Fast58:136pass/1fail/300subtests, fixture mock Git precedente da adeguare; fixture corretta senza indebolire il gate reale. Nessun PASS suite corrente dedotto da questi risultati.

Ingresso core3940.5060664880657s. La ripresa conserva la contabilità completa; dopo il blocco storage l'intervallo di attesa approvazione/Docker separato non è workload core.180s conservativi coprono preparazione al confine; nuovo gate real-time dalla ripresa, senza sommare intervalli annidati. Ultimo cumulativo addebitato `7194.886280257022`s, residuo `5.113719742977992`s; monitor dev59 senza stop esterno, exit `2`; child `timed_out=true`, exit `-15`, durata0.42647469812072814s perché il gate sul residuo non ammetteva una deadline sufficiente. Il FAIL è il timeout interno effettivo dovuto al residuo, non un rifiuto sandbox o del monitor storage. Il driver non aumenta7200s: child entro residuo, outer child+180 e monitor sempre entro residuo totale.

Quota spazio estesa esplicitamente dall'utente di48MiB:352MiB cumulativi, stessi Hentry/pool/stop/riserva. Misura finale H `890355712`, Delta `336814080`, residuo `32284672` byte. Le fixture storiche pytest6/7 da15MB sono conservate. Nuovi tmp sono esclusivi e raccolgono solo PASS; nessun cleanup globale.

Tutte le raccolte processi proprie senza superstiti. Paid0, rete nuova0; nessun Git write, agenti, servizio normale, inferenza/GPU, pesi/font remoti o modifica host. Baseline62pass5fail/perdite e FAIL storici restano invariati.

## Unico confine ancora necessario

Estensione **+900s core**, totale8100s; nessuna ulteriore quota storage/Docker/rete. Necessaria per dev/API I/E correnti, C-fast ufficiale e standalone reale/negativi/run_id, packaging/API/discovery, V1 completa con outer/postcheck e consegna finale. Il limite7200s resta in vigore finché non autorizzata. Non è una consegna per il test rosso: il fix è pronto e la prosecuzione è fermata dal gate reale di tempo.

A7/review/arbitrato al supervisore; V10/V11 e inferenza escluse. V1 e le prove host invalidate restano aperte. CLI/fedeltà/baseline storiche preservate, nessun PASS trasferito prima del manifest finale delle equivalenze.

Docker finale: H upper15.861.197.984 byte, residuo1.318.671.200 sul pool16GiB; cumulativo conservativo872.2351763881743s/7200, network upper496.040.215 byte storico, nessuna nuova rete/wire non misurato. Dettaglio native Shared/immagini/record reviewer nel ledger Docker finale.
