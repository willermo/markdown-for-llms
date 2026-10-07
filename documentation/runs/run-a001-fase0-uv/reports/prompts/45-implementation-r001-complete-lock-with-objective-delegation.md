# Implementazione r001 — completare il lock con delega per obiettivo

Agisci come **implementatore** in `/home/davide/workarea/markdown-for-llms`;
prosegui nella stessa chat. Risultato: acquisire i metadata necessari dalle
fonti già configurate, produrre il lock universale originale e verificarlo
offline. Preparazione, diagnosi, scelta della soluzione e fix sono inclusi.
Non tornare al supervisore per la sola assenza di un caso o comando nel prompt.
Nessuna delega, commit/merge/push/deploy/cleanup.

Leggi AGENTS, skill manage-implementation-run, protocollo corrente,
STATE/HANDOVER, checkpoint autore/report-r010 e checkpoint supervisore
`temp/run-a001-fase0-uv/handovers/supervisor-objective-delegation-r001.md`.
Nella run leggi `arbitrations/addendum-operational-protocol-r013.md` e
`implementation/stages/impl-r001-stage-package-s015/request.json`; sotto
`evidence/supervisor-implementation-r001/impl-r001-stage-package-s015/` leggi authorized-scope,
source-policy, reception, transition, checks e freeze-verify. Scope completo
contiene template argv/cwd/env, input e limiti; non ricostruirli dalla memoria.
R013 prevale sulle vecchie restrizioni host e liste esaustive di mezzi.

Piano r003 SHA462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b,
arbitrato SHA f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d
invariati; GO piano soltanto, NO_GO storici conservati. Feature/run-a001-uv,
HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091, indice vuoto.

S015: manifest147723byte/SHA0b52990087f8e6f5e8fe64b86ab563267c3e41a8736f2e78babf939fca89c7d7,
worktreeebc49cba700f462eca438220587c552e3778a2815bb20acd4359eb6680c4ad12, 123file/457artefatti.
RequestSHAa7149442ae0f5c10bdbdbe13d772b8da44506f9ace425b89bd88edff56dcb289; scopeSHA4b9701b0f6cfcee41ea349221aec3be074b024cbd281b534c2a140d423a76999.
Ricalcola byte/hash e prima/dopo esegui da radice:
`python3 -B scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-package-s015`.
S014 storico per nove delta governance dichiarati; input prodotto/helper e
artefatti precedenti intatti. Non chiamare MATCH il vecchio worktree con nuovi
file. Nessuna auto-modifica snapshot/input confronto né PASS trasferito.

## Decidi, verifica e continua

I mezzi reversibili non enumerati sono delegati entro obiettivo e confini.
Scegli strategie diagnostiche, endpoint delle fonti configurate, launcher/reader,
output esclusivi, log e timeout nei limiti. Documenta nel report le decisioni
significative e le verifiche invalidate/ripetute; le due review le valuteranno.
Non aprire una preparazione o un handoff per ogni correzione. Prima di fermarti
indica il requisito o limite concreto che impedisce di continuare autonomamente.

44 ha corretto pytest/colorama nella stessa chat; cache140nomi/26nuovi/849record,
due priming exit0 e due R/D4 PASS. Lock assente per metadata torch CPU.
L'indice CPU era già previsto: la whitelist PyPI precedente era incompleta.
Sono coperte tutte le fonti pubbliche già configurate, PyPI e indici originali
CPU/cu126 PyTorch, e loro endpoint/redirect con provenienza HTTPS verificabile.
`download-r2.pytorch.org` osservato per cu126 non dimostra il redirect CPU.
Se uv non espone la catena, usa una lettura pubblica mirata verificabile;
non TLS bypass, credenziali o invio documenti. Nuovo host di trasporto verificato
non richiede mandato; nuova fonte logica non dimostrata resta fuori perimetro.

Puoi partire dal template ricevuto torch==2.7.1 diagnostico universale con
--index CPU originale, PyPI default, --only-binary=:all:/no-sources/config vuota,
stesso uv0.10.10/managed CPython3.12.13/env chiuso. Source policy global_no_build
è divieto di eseguire backend, non invito ad aggiungere --no-build incompatibile
con only-binary in questa CLI. Nessun --no-deps, pin prodotto da PyPI,
override grafo o restrizione delle piattaforme per ottenere PASS.

Crea nuove evidenze `evidence/implementation-r001/resume-package-s015/`, riusa
work/cache senza cleanup; adatta propri strumenti non congelati, template e
suffissi entro scope. Argv strutturati shell=False/close_fds=True, env puro
chiuso19 per uv e16 R, nessun merge con os.environ. Log normali senza verbose/
quiet; wrapper e runner congelati invariati. Conserva errori/diagnosi/actual
argv e ricevute; retry soltanto dopo diagnosi, raccolta processi e gate valido.

Acquisizione metadata→audit/provenienza/sigillo cache→divieti build per tutti
i nomi reali salvo eccezione EbookLib0.18→lock universale originale offline
sotto nuovo R/Firejail net=none/D4 sul medesimo comando→audit lock→check offline
con nuovo R/D4 e hash lock intatto. Altri miss nello stesso obiettivo si risolvono
nella stessa chat. Chiusura64nomi nuovi dai114iniziali,26usati/38restanti;
cache nativa autentica, niente cache forgiata/lock manuale. Baseline protetta,
fonti/pin/extras/config/env/grafo prodotto invariati. Nessuna modifica tracciata
in questa tranche; se una scelta lo richiede fuori scope, consegna scostamento
concreto anziché attribuire una prova ufficiale a input modificati.

Budget **stesso32MiB cumulativo**, Hentry525762560; non32 nuovi. H=max(logical,
allocated) di run+.venv-python via lstat senza seguire link, Delta=max(0,H-Hentry),
Delta<33554432 e H+max(0,33554432-Delta)+16777216<939524096. Conta vecchi costi,
nuovi registri/script/snapshot; categorie condivise non additive. Circa8–9MiB
residui da misurare; stima metadataCPU2MiB+registri1MiB non garanzia. Pool1GiB,
stop896/libero1GiB/riserva esterna16MiB, file32MiB/JSON8MiB/stream1MiB invariati.
Monitor0,5s non quota atomica. Tempi cumulativi residui metadata897,7209697877988s,
lock899,2269420649391s/check120s; sottrai consumi, outer=figlio+180. Non ricopiare
storia/cache a ogni tentativo; sigilli compressi e riferimenti a dati intatti.

Niente backend/install/sdist/Git/pesi/font/archivi ML completi. Per metadata
pesanti usa PEP658/range o mezzo pertinente ammesso; rischio residuo non atomico
ricevuto, mancanza di un hard cap wire non blocca da sola il tentativo. Se
osservi fallback a payload pesante vietato, interrompilo e cerca un mezzo
ammesso nella stessa chat. Non dichiarare URL/body/redirect osservati quando
sono NOT_MEASURED. Nessuna modifica host/privilegi/socket/profili/rete.
Costo oltre limite, requisito/architettura nuovi, origine/confinamento/privacy
non rispettabili, input confronto alterati, rifiuto sandbox o impossibilità
dimostrata sono escalation; fermare il solo lavoro dipendente.

Consegna delivery, `implementation/report-r011.md`, completion nello stage s015
con esiti reali separati e identità request/scope/manifest, decisioni, risorse/
tempi/processi e limiti. Aggiorna soltanto checkpoint/report autore con ingressi
preservati; registri comuni del supervisore. Stato WAITING_FOR_SUPERVISOR_RECEPTION,
prossimo supervisore05+r013. Se lock/check pronti, includi richiesta concreta
per promozione/S-B-I-E (input/comandi/output/costi); quel seguito non si esegue qui.
Baseline62pass5fail/perdite e altri FAIL/lacune storici preservati; S-B-I-E/V,
V7/V8 costi distinti, V10V11 esclusi, due review indipendenti/GO finale aperti.
Nessun servizio automatico da attendere.
