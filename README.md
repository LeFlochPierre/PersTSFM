# PersTSFM

## Research question

When should a pretrained time-series model begin to personalize to a previously unseen entity as entity-specific observations accumulate?

## Initial hypothesis

For a new entity:

- with little entity-specific data, the population/global model should perform best;
- as observations accumulate, a personalized model should improve;
- there may be a crossover point n* at which personalization becomes beneficial;
- n* should depend on entity heterogeneity, observation noise and representation mismatch.

## Phase 1

Test whether a personalization crossover exists empirically.

### Dataset
ElectricityLoadDiagrams20112014

Entity:
individual electricity customer

### Initial models

1. Global / zero-shot time-series foundation model
2. Local model trained only on the held-out entity
3. Global model + simple personalized residual correction

### Evaluation

Hold out complete entities.

For each test entity, reveal increasing amounts of history:

n = 0, 2, 4, 8, 16, 32, 64 windows

At each n:
- adapt using past data only;
- forecast the future;
- record forecasting loss.

Main output:

forecast error vs number of entity-specific observations.

## Key question

Does there exist a systematic crossover point n* where personalized prediction becomes better than population prediction?