# R010 — correzioni operative delegate e ripresa del lock

2026-10-06. Disposizione del supervisore su richiesta esplicita dell’utente.
Piano/arbitrato r003 e r009 restano invariati nei requisiti; nessun GO finale.

## Causa e variazione

Mandato41: il supervisore aveva imposto argv900 e immutabilità; il helper
rifiutava valori superiori a120. Il CLI FAIL/exit2 precede R/uv/lock e non è
un timeout reale. L’implementatore ha rispettato il vincolo errato. Il costo
superfluo del passaggio è attribuito al mandato, non alla mancata correzione.

Recepita direttamente la patch già consegnata dall’autore: soltanto confronto
CLI120→900 e messaggio120→900, default120 invariato. Nessuna modifica a R,
Firejail, monitor, ambiente, output, gate o policy native. Nuova identità del
helper nel freeze s012; nessun PASS precedente trasferito alla versione nuova.

R010 supera espressamente «argv/env/target immutabili» e «nessun helper
modificato» di41 quanto alla delega operativa descritta qui e nella scope s012.
Gli artefatti s011 restano immutabili come documenti storici. Gli argv in scope
sono template ricevuti: conservare quello storico e registrare quello eseguito.

## Autonomia effettiva

Correggere e completare lock/check nella stessa chat. Sono delegati senza nuova
supervisione: deadline compatibili (lock massimo900s, check massimo120s), tempi
esteriori sufficienti alle guardie (deadline+180s), suffissi esclusivi degli
output nelle evidenze ammesse e correzioni di launcher/reader non congelati.
Non serve nuovo freeze per queste scelte quando gli input della prova restano
identici. Se un tool rifiuta un parametro, correggere dopo diagnosi e proseguire.
Un timeout effettivo richiede fine dei processi propri, parziali preservati,
gate validi e verifica della ripetibilità prima del nuovo tentativo. Tempi
workload cumulativi ammessi per attività: lock900s, check120s; le invocazioni
respinte prima del workload non consumano la deadline del workload. Nessun
retry cieco o numero arbitrario di un solo tentativo per gli errori ordinari.

Codice/helper che determina la prova: correggerlo e verificarlo nel perimetro
nella stessa chat, senza chiedere prima il permesso di correggere; le prove
invalidate non si trasferiscono. Se serve un nuovo freeze ufficiale, consegnare
insieme correzione completa, verifiche pertinenti e input pronti. Per questo
seguito il wrapper corretto è già ricevuto e congelato; altre correzioni possono
essere verificate preliminarmente ma non fingere MATCH o R ufficiale del nuovo
helper rispetto al freeze s012. Nessun auto-freeze del supervisore.

Restano arresti sostanziali privacy/isolamento/integrità/input di confronto,
fonti/pin/grafo/metadati nuovi, budget non coperto e rifiuto sandbox. Fermare
l’attività dipendente; continuare lavoro indipendente ammesso. Gli esiti CLI
ordinari, per se stessi, non sono motivo di consegna o richiesta di permesso.

## Perimetro e risorse ereditati

Solo lock/check offline nella copia già identificata, R/D4/Firejail net=none,
114 divieti package e gatecache chiusa, EbookLib0.18 metadata osservati. Nessun
backend, build, install, acquisizione, fonte nuova o promozione al prodotto.
Scope s012 conserva native argv/env/target della s011, baseline e managed noti.
Le directory work/tmp/config già esistenti si verificano e si riusano: non
applicare il vecchio requisito di crearle come se non esistessero.

Continua la medesima tranche16MiB della s011, Hentry525762560byte, con caps
8attività/8registri,16MiB esterni e pool1GiB/stop896MiB. Non accogliere il
nuovo budget proposto dall’autore. Le nuove evidenze supervisore e snapshot
consumano il residuo; niente cleanup o ledger esterno. Zero nuovi costi remoti.

Consegna report-r008, completion s012 ed evidenze nuove con risultati reali e
lineage alla CLI failure s011. Non riscrivere report/receipt s011. Se lock/check
riescono consegnare insieme la richiesta concreta per il seguito S/B/I/E;
nessun altro planning-only per il lock. Product/review/arbitrato restano aperti.
