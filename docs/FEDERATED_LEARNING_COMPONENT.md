# Federated Learning Component: U-FIN Multi-Institution Evaluation

**Document Version:** 1.0  
**Date:** May 2026  
**Status:** Final paper result  
**Canonical Results:** `results/federated/exp1_final/`

---

## Executive Summary

This document describes the federated learning component of U-FIN. The goal is to evaluate whether the full fraud-detection pipeline can benefit from cross-institutional model sharing without pooling raw transaction records.

Because the IEEE-CIS Fraud Detection dataset does not contain real institution identifiers, the experiment uses a **simulated multi-institutional split**. Unique `card1` values are partitioned into three disjoint clients, so transactions sharing the same card identifier remain within one simulated institution. This creates a card-based non-IID client partition while preserving a common held-out validation set for controlled comparison.

**Best Federated Result:** FedAvg+FedBN reaches AUC 0.9290 at round 146.  
**Main Comparison:** FedAvg+FedBN improves over local-only training by 12.45 AUC points and retains 96.39% of centralized U-FIN AUC while sharing only model updates.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Federated System Overview](#federated-system-overview)
3. [Client Partitioning](#client-partitioning)
4. [Federated Model Components](#federated-model-components)
5. [Training Configuration](#training-configuration)
6. [FedAvg+FedBN Aggregation](#fedavgfedbn-aggregation)
7. [Results](#results)
8. [Communication Cost](#communication-cost)
9. [Artifacts](#artifacts)
10. [Limitations](#limitations)

---

## Motivation

Financial organizations often cannot pool raw transaction data because of competitive sensitivities and regulatory constraints. Federated learning allows each client to train locally and share only model updates with a central server. In this project, the federated layer is used to test whether cross-client learning improves fraud detection compared with isolated local-only training.

The centralized U-FIN model remains the non-private upper bound because it trains on pooled raw data. The federated result is therefore evaluated as a privacy-utility trade-off rather than as a replacement for centralized training.

---

## Federated System Overview

```
Client A: NationalBank_A       Client B: RegionalBank_B       Client C: FintechPay_C
        │                              │                              │
        ▼                              ▼                              ▼
  Local U-FIN training            Local U-FIN training            Local U-FIN training
  TCN + GAT + CaT-GNN             TCN + GAT + CaT-GNN             TCN + GAT + CaT-GNN
  + DAE + GatedFusion             + DAE + GatedFusion             + DAE + GatedFusion
        │                              │                              │
        └────────────── model updates only ───────────────┬───────────┘
                                                          ▼
                                                FedAvg+FedBN Server
                                                          │
                                                          ▼
                                                 Updated global model
```

Only model updates are shared with the server. Raw transaction records remain local to the simulated clients.

---

## Client Partitioning

The partition is based on unique `card1` values. This choice keeps transactions associated with the same card identifier inside the same simulated institution and creates realistic heterogeneity across clients.

| Client | Samples | Fraud Rate | Local-only AUC | Local-only F1 |
|---|---:|---:|---:|---:|
| NationalBank_A | 176,982 | 3.77% | 0.8081 | 0.2904 |
| RegionalBank_B | 147,182 | 3.23% | 0.8018 | 0.2694 |
| FintechPay_C | 148,268 | 3.44% | 0.8035 | 0.2682 |
| Mean | --- | --- | 0.8044 | 0.2760 |

**Validation protocol:** all clients are evaluated on the same held-out validation set of 118,108 samples. This makes centralized, federated, and local-only results directly comparable.

---

## Federated Model Components

The federated experiment trains the full U-FIN neural pipeline:

- `FraudTCN`
- `FraudGAT`
- `CaT_GNN`
- `ImprovedDAE`
- `GatedFusion`

The XGBoost meta-learner is part of the centralized final classifier, while the federated experiment focuses on the neural representation pipeline and gated classifier trained through model-update aggregation.

---

## Training Configuration

Key configuration from `results/federated/exp1_final/fedavg/metrics/summary.json`:

| Parameter | Value |
|---|---:|
| Algorithm | FedAvg |
| FedBN enabled | true |
| Total logged rounds | 146 |
| Local epochs | 2 |
| Batch size | 1024 |
| Client fraction | 1.0 |
| Minimum clients per round | 3 |
| Initial LR | 0.001 |
| Final LR | 1e-6 |
| LR patience | 5 |
| LR factor | 0.5 |
| Positive class weight | 27.0 |
| Max gradient norm | 1.0 |
| Input dimension | 263 |
| Embedding dimension | 64 |
| DAE mask ratio | 0.3 |

Although the notebook configuration originally allowed multiple FL algorithms, the canonical final result set uses FedAvg+FedBN.

---

## FedAvg+FedBN Aggregation

FedAvg performs sample-weighted aggregation of shared model parameters:

```
theta_global = sum_k (n_k / n_total) * theta_k
```

FedBN modifies this by keeping BatchNorm statistics local to each client. This is useful in the card-based non-IID split because client transaction distributions differ. The shared layers benefit from cross-client aggregation, while BatchNorm statistics remain adapted to each local distribution.

In practical terms:

- Shared encoder and classifier weights are aggregated.
- BatchNorm running statistics remain client-local.
- The server receives model updates only.
- Raw transaction records are not pooled.

---

## Results

### Main System Comparison

| Setting | AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Centralized U-FIN | 0.9637 | 0.7880 | 0.8568 | 0.7295 |
| FedAvg+FedBN | 0.9290 | 0.5966 | 0.7118 | 0.5134 |
| Local-only mean | 0.8044 | 0.2760 | 0.2630 | 0.2911 |

### Interpretation

The centralized model remains the expected upper bound because it trains on pooled raw data. FedAvg+FedBN incurs a 3.48 AUC-point loss relative to centralized pooling, but improves over isolated local-only training by 12.45 AUC points.

The federated model:

- retains 96.39% of centralized AUC,
- closes 78.17% of the local-to-centralized AUC gap,
- reaches best AUC 0.9290 at round 146,
- shares only model updates rather than pooled raw data.

This supports the central architectural claim that cross-institutional model sharing can improve fraud detection under privacy constraints, with a clear utility trade-off relative to fully centralized training.

---

## Communication Cost

From `results/federated/exp1_final/metrics/communication_cost.json`:

| Metric | Value |
|---|---:|
| Model size | 5.28 MB |
| Total parameters | 1,320,016 |
| Upload | 2,312.67 MB |
| Download | 2,312.67 MB |
| Total communication | 4,625.34 MB |
| Logged rounds | 146 |

The communication estimate reflects model-update exchange over the logged FedAvg+FedBN rounds.

---

## Convergence

From `results/federated/exp1_final/metrics/convergence_analysis.json`:

| Target AUC | Round Reached |
|---|---:|
| 0.80 | 1 |
| 0.82 | 3 |
| 0.84 | 6 |
| 0.86 | 9 |
| 0.90 | 36 |
| Peak 0.9290 | 146 |

Average round time was approximately 85.77 seconds.

---

## Artifacts

| Path | Description |
|---|---|
| `notebooks/final/experiment_federated.ipynb` | Final federated experiment notebook |
| `models/federated/fedavg_fedbn_best/best_pipeline.pt` | Canonical best FedAvg+FedBN model |
| `results/federated/exp1_final/metrics/fl_final_metrics.json` | Canonical final FL metrics |
| `results/federated/exp1_final/fedavg/metrics/summary.json` | FedAvg+FedBN summary and config |
| `results/federated/exp1_final/fedavg/histories/round_history.json` | Per-round metric history |
| `results/federated/exp1_final/local_only/metrics/summary.json` | True local-only baseline |
| `results/federated/exp1_final/tables/table1_comparison.json` | Algorithm-level comparison |
| `results/federated/exp1_final/metrics/communication_cost.json` | Communication cost summary |
| `results/federated/exp1_final/metrics/convergence_analysis.json` | Convergence summary |
| `results/federated/exp1_final/figures/fig2_convergence.png` | FedAvg+FedBN convergence figure |

---

## Limitations

- The institution split is simulated because IEEE-CIS does not provide real institution identifiers.
- The split should be described as a simulated multi-institutional or card-based non-IID partition, not a real bank deployment.
- The experiment shows that raw data are not pooled and only model updates are shared; it does not claim formal privacy guarantees such as differential privacy.
- Centralized U-FIN remains the non-private upper bound.

---

## Summary

The federated component validates the architectural claim that U-FIN can benefit from cross-client learning under non-IID transaction distributions. FedAvg+FedBN substantially improves over isolated local-only training while preserving a clear utility trade-off relative to centralized pooled-data training.
