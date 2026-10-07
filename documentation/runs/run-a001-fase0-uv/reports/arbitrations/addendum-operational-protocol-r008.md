# Addendum operativo r008 — metadati EbookLib osservati per il lock no-build

2026-10-06. Supervisore, ricezione mandato39; stesso obiettivo36 e GO piano
r003. Piano/arbitrato immutati; errata operativa verificata, nessun GO codice.

## Ricezione e limiti

Le due build offline della sola EbookLib0.18 sono realmente concluse exit0,
backend setuptools.build_meta:__legacy__84 osservato, R/wrapper e namespace
verificati. Ricezione propria:68 file referenziati ricalcolati,15 membri wheel,
RECORD e9 moduli/2 licenze confrontati con l'archivio auditato. Payload e byte
compressi coincidono; timestamp ZIP differiscono e gli hash esterni differiscono.
**Riproducibilità byte FAIL**, conservata; nessuna normalizzazione degli archivi.

L'enumerazione separata dello startup bootstrap prima di ciascuna build manca.
Non attesto il gate completo pre/post, non ricostruisco la prova retroattivamente.
I metadati osservati sono anche coerenti con il setup statico già ricevuto:
Name EbookLib,Version0.18,Requires-Dist lxml/six, nessun Requires-Python/extra.
Questi dati sono utilizzabili con provenienza e limite espliciti; non trasferisco
PASS a prodotto, ABI o installazione. Per i prossimi avvii lo startup si controlla
e registra prima e dopo, senza ripetere queste due build o la baseline acquisita.

## Scelta concreta e alternativa esclusa

Non accolgo il prime registry proposto: richiederebbe esecuzione del backend con
accesso HTTPS, non coperta dal wrapper net=none, e riuso della cache incerto.
La proposta d'autore rimane immutata/NOT_AUTHORIZED.

Adotto il meccanismo ufficiale `[[tool.uv.dependency-metadata]]`, confermato
nel documento e schema **uv0.10.10** ricevuti dal supervisore. Fornisce metadati
espliciti al resolver evitando la build. Unica dichiarazione ammessa:

```toml
[[tool.uv.dependency-metadata]]
name = "ebooklib"
version = "0.18"
requires-dist = ["lxml", "six"]
```

Omissione Requires-Python ed extra rispecchia i metadati osservati, non inventa
vincoli. Versione obbligatoriamente circoscritta0.18; nessun override di versioni,
dipendenze escluse, grafi ridotti, indici sostituiti o fonte wheel locale.
La fonte resta registry originale; il lock deve contenere sdist con SHA pubblico
38562643a7bc94d9bf56e9930b4927e4e93b5d1d0917f697a6454db5a1c1a533.
Questo è un input dichiarato/proveniente, distinto da cache o lock fabbricati.
Il divieto antecedente di metadata manuali non impedisce questa nuova eccezione
esplicita, verificata e congelata. Vale soltanto per questo package/versione.

Il supervisore predispone e congela una **nuova copia di input** con quel solo
delta TOML. Nessun codice prodotto eseguito/modificato in ricezione; root pyproject
e vecchia lock-project-r001 rimangono intatti. La promozione nel prodotto avverrà
solo dopo ricezione del lock e con nuova identità degli input pertinenti.

## Mandato per risultato, senza preparazione intermedia

Package-s010/scope identificano tre comandi nativi pronti: lock offline/no-build;
se fallisce **soltanto per metadata mancanti in cache**, un lock online/no-build
distinto; dopo un lock exit0, check offline/no-build con hash invariato.
Non è retry cieco: offlineFAIL resta conservato e il secondo ramo è un'acquisizione
di metadata esplicitamente ammessa, già prevista dal mandato37/r005, con nuovi log.
No-build rimane su tutti i comandi. Nessuna build, installazione, import app/native,
dry-run install, prime registry, nuovo Python o backend.

HTTPS soltanto indici originali PyPI e PyTorch CPU/cu126 e loro normali endpoint
pubblici di metadata/artifact; ambiente chiuso senza proxy/auth ereditati, retries0
e download concorrenti1. PEP658/range e piccole wheel per metadata sono ammessi;
fallback a payload ML/native pesante, requisito dinamico o altro backend ⇒ STOP.
Non chiamare questo ramo net=none/R PASS: è il resolver no-build con accesso HTTPS
già circoscritto in r005, non una prova applicativa/backend. I probe nativi del
solo interprete managed già verificato sono ammessi; nessun figlio applicativo.
D4 e isolamento completo restano obbligatori nelle future prove R→S/B/I/E.

Nuova tranche32MiB incrementali (24 attività/cache/copied input,8 registri)
più16MiB esterni; ledger run+.venv-python max1GiB/stop896MiB/libero1GiB,
directory/link inclusi senza seguirli. La quota precedente è conclusa, non sommata.
RLIMIT_FSIZE32MiB/file,log1MiB/stream,JSON8MiB,monitor0,5s/gap target1s.
Lock child900s/outer1080s, check120s/outer300s; raccogli sessioni vive.
Stima body metadata≤24MiB, wire non misurato: CLI non offre quota HTTP atomica;
monitor storage e cap file non sono una garanzia sui byte di rete. Nessun cleanup
per superare gate. Superamento di cap/integrità o scostamento sostanziale ⇒ STOP.

Fix ordinari dei launcher/reader non congelati nella stessa chat, conservando
esiti e prove. Input congelati non si cambiano; no auto-freeze, helper invariati.
Consegna lock/check reali, audit grafo/fonti/hash e richiesta concreta dei successivi
S/B/I/E/costi; non soltanto preparazione. Nessun PASS anticipato se la cache non
basta o una seconda dipendenza richiede metadata non ricevuti.

Baseline originale già acquisita, suite62pass5fail e perdite preservate. V7/V8
obbligatorie con costi distinti; V10/V11/pesi/font/inferenza esclusi. Due review
indipendenti e arbitrato finale necessari. Git manuale, nessun commit/merge/push/
deploy/cleanup o servizio automatico da attendere.

Fonti: [risoluzione uv pinned](https://github.com/astral-sh/uv/blob/0.10.10/docs/concepts/resolution.md#dependency-metadata),
[schema pinned](https://github.com/astral-sh/uv/blob/0.10.10/uv.schema.json),
[ricevute locali — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
