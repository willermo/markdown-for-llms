---
name: write-diataxis-docs
description: "Scrive o riorganizza README e documentazione operativa in docs/ secondo Diátaxis, per utenti e sviluppatori del convertitore. Usare per tutorial, guide pratiche, reference e spiegazioni; per proposte e scelte architetturali usare invece il registro ADR in documentation/."
---

# Scrivere documentazione secondo Diátaxis

1. Leggere [indice docs](../../../docs/README.md), pagina interessata e comportamento
   del codice pertinente. Consultare [ADR 0005](../../../documentation/decisions/0005-documentation-and-governance.md)
   per la separazione tra progetto e istruzioni d'uso.
2. Individuare il bisogno principale del lettore e scegliere una sola forma:
   - `docs/tutorials/`: apprendimento guidato, prerequisiti, campione e risultato atteso;
   - `docs/how-to/`: compito concreto, passi essenziali e verifica finale;
   - `docs/reference/`: dettagli esatti di opzioni, schemi, default, vincoli e compatibilità;
   - `docs/explanation/`: concetti e motivazioni, con collegamenti agli ADR pertinenti.
3. Separare bisogni diversi in pagine collegate quando la pagina diventerebbe confusa.
   Indicare pubblico e versione quando rilevanti; non creare alberi duplicati per
   utenti e sviluppatori con gli stessi contenuti.
4. Seguire il [workflow documentazione](../../workflows/documentation.md). Verificare
   comandi ed esempi sul prodotto disponibile; non inventare API, variabili `.env`
   o procedure di installazione per funzionalità ancora soltanto pianificate.
5. Aggiornare navigazione e collegamenti. Il README introduce il progetto e indirizza
   alle guide; non deve diventare una seconda copia dell'intera reference.

Consegnare pagine nella collocazione corretta e indicare gli esempi verificati.
Per scelte sul metodo consultare la [fonte Diátaxis](https://diataxis.fr/), evitando
di trasformare i quattro tipi in un semplice requisito di quattro cartelle vuote.
