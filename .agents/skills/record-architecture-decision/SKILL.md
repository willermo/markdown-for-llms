---
name: record-architecture-decision
description: "Registra o aggiorna decisioni architetturali di questo convertitore in documentation/, mantenendo coerenti ADR, questioni aperte e roadmap. Usare per scelte di motori, modello documentale, API, storage, frontend o distribuzione; non per piccoli refactoring senza conseguenze architetturali."
---

# Registrare una decisione architetturale

1. Leggere il [registro ADR](../../../documentation/decisions/README.md) e gli ADR
   pertinenti. Distinguere la richiesta attuale, le scelte già concordate e le ipotesi.
2. Seguire il [workflow architetturale](../../workflows/architecture-change.md).
   Riutilizzare il [template](../../../documentation/decisions/template.md) per una
   nuova decisione significativa; aggiornare una proposta esistente se è lo stesso tema.
3. Motivare la scelta con vincoli del convertitore: fedeltà, provenienza, CPU/GPU,
   uso locale/remoto, manutenzione e verificabilità. Citare fonti primarie aggiornate
   quando la decisione dipende da capacità, licenze o versioni di strumenti.
4. Riportare separatamente stato della decisione e dell'implementazione. Registrare
   come accettata una scelta già autorizzata dalla conversazione; non chiedere di
   approvarla di nuovo. Se manca una decisione sostanziale, registrare una proposta
   esplicita e il punto da risolvere, continuando il lavoro indipendente.
5. Aggiornare indice, roadmap e questioni aperte senza riscrivere il draft storico.
   Se una scelta accettata viene sostituita, collegare vecchio e nuovo ADR.

Consegnare ADR e collegamenti coerenti, con alternative, conseguenze e criterio di
verifica. Non trasformare automaticamente l'ADR in un'implementazione o un merge.
