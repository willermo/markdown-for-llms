# Censimento r013 prima del codice

2026-10-11, implementatore, mandato43. Ingresso MATCH; V12 preservato in
work/implementation-r013/r012-observations.py (SHA 5cabe677…bc22).

| Fonte/callsite V12 | Dominio, ordine e disponibilità raw | Proprietà da provare |
| --- | --- | --- |
| attrs442 → peel_attrs452 → heading1120 | Metadata finale, chiamato prima del testo; posizione del gruppo nel raw disponibile | Prima causa secondo char sorgente, attributo escluso dai valori sicuri, coda visitata |
| attrs → div1077 | Opener fisico prima del corpo, anche annidato/oltre limite | Non interrompe domini seguenti; controllo anche quando il div non è proiettabile |
| attrs → fence1100 | Info prima del corpo letterale | Guardia metadata distinta da corpo inerte e ownership del fence |
| attrs → figure1145 | Caption costruita prima dei metadata finali | Raw e ordine caption/metadata/coda; più cause |
| attrs → inline code668/link695/span708 | Metadati nel dominio trasformato, scanner HTML passa prima della proiezione | Eventi metadata già nel ledger, posizione da source_map; non ordine delle due passate |
| inline_html_events → validate_html | Aperture reali, raw inline esatto e raw block container | Una visita per apertura sorgente, ordine/multiplicità, incluso dopo errori |
| validate_html → HTMLGuard → PassiveHTML | Callback per ogni apertura; posizione locale getpos e snippet disponibili | Tutte le aperture visitate anche con più violazioni nello stesso dominio |
| HTMLTree → PassiveHTML/report_unsafe | Duplicazione della guardia durante proiezione; raise ValueError per duplicati/depth/nodi | Albero non è fonte indipendente; riuso guardia del dominio, errore non interrompe guardie |
| inline raw HTML668 / html_nodes820 | Feed aggiuntivo del payload/fragment già guardato | Evitare duplicati; standalone conserva validazione |
| html_nodes catch Unsafe/ValueError | Guardia precede costruzione albero, fallback unknown | Nessun catch tronca la visita del resto del documento |
| inline HTML/fallback e inline_delimiter | Parser del frammento chiuso, ValueError catturato | Guardia ledger precedente; residui espliciti, code/math successivi inerti F44 |
| _observe raw fence HTML/LaTeX, HTML blocco e relativo catch | Payload fisico, cstart/body_start conosciuti | Trasporto assoluto, ordine tra domini e guardia indipendente dalla proiezione |
| node fallback unknown1051 | Raw fisico; solo senza coverage e senza unknown | Fonte di guardia con posizione; nessun secondo feed sulla coda già coperta |
| observe987/finally | Rilancio prima causa, reset | Sessione chiusa anche per UTF8/limite/errore; chiamata successiva pulita |
| PassiveHTML adapter | Tag non ammesso, attr fuori subset/attivo, meta, URI assente/vietata/encoded/evasiva | Politica esistente invariata; nessun accesso URI, diagnosi sorgente |
| audit / observe_bundle / CLI | Catch Unsafe dopo observe; path Markdown verificati dopo reader | Limite A8 esplicito; nessun ordinamento reader+audit nuovo |

Scelta: raccolta di eventi delle guardie con coordinate sorgente nel reader
esistente, drenaggio ordinato indipendente dalla proiezione. Gli attributi
attivi sono diagnosticati ed esclusi dai valori restituiti, mai accettati.
Le chiamate ripetute sul medesimo dominio non duplicano aperture. Nessun
nuovo modulo/schema pubblico. Coverage resta privata fino al consumo node;
la pulizia JSON avviene al termine dell'osservazione.

F55: una decisione riusabile di fence considera inizio/separazione/paragrafo,
marker, metadata e chiusura. La classe ordinaria EOF non ha ownership code.
Gli AST Pandoc locali decideranno le premesse sintattiche prima del runtime
V13; contenuto/asset/sicurezza restano attesi manuali. Proteggere raw esatto
HTML/LaTeX, source-map, LF label, F36 e token opachi. Matrice e risultati
verranno legati a ogni riga nel report finale; questa nota non attesta PASS.

## Chiusura dell’inventario

Il [report permanente r013](../../../../benchmarks/2026-10-11-a002-r013-guardia-e-fence.md#inventario--proprietà--prove)
e i relativi dati enumerano i casi e i risultati per ogni callsite censito,
raggruppando gli adapter con lo stesso contratto. Risultato autore: 285/285
più16/16 metadata LF, 199 AST principali più16 AST attrs, due controprove
0/3; non è una review o un GO. Il confronto
reader/audit resta esplicitamente limitato; l’ordine globale fra i due non
è stato esteso. Le righe iniziali di questa nota rimangono il censimento
scritto prima del codice, non una ricostruzione retroattiva.
