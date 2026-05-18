# Results Directory

This folder stores evaluation outputs, metric summaries, plots, histories, tables, and generated arrays. Trained model binaries and fitted scalers are stored separately under `models/`.

All paths below are relative to the repository root, `IES_Challenge_UFIN/`.

## Structure

```text
results/
├── centralized/
│   ├── tcn_results/
│   │   ├── figures/
│   │   ├── metrics/
│   │   └── histories/
│   ├── gat_results/
│   │   ├── figures/
│   │   └── metrics/
│   ├── catgnn_results/
│   │   ├── figures/
│   │   ├── metrics/
│   │   └── histories/
│   ├── dae_results/
│   │   ├── figures/
│   │   ├── metrics/
│   │   └── arrays/
│   └── results_final_fusion_valSet/
│       ├── figures/
│       └── metrics/
├── federated/
│   ├── exp1_final/
│   │   ├── figures/
│   │   ├── metrics/
│   │   ├── tables/
│   │   ├── fedavg/
│   │   │   ├── metrics/
│   │   │   └── histories/
│   │   └── local_only/
│   │       └── metrics/
│   ├── legacy_algorithms/
│   │   └── fed_results/
└── rlhf/
    ├── arrays/
    ├── figures/
    ├── histories/
    ├── metrics/
    └── Readme.md
```

## Canonical Result Files

- Centralized U-FIN: `results/centralized/results_final_fusion_valSet/metrics/fusion_metrics.json`
- Federated FedAvg+FedBN: `results/federated/exp1_final/metrics/fl_final_metrics.json`
- Local-only baseline: `results/federated/exp1_final/local_only/metrics/summary.json`
- DDQN/static threshold comparison: `results/rlhf/metrics/static_threshold_comparison.json`

## Model Files

All model artifacts have been moved to `models/`:

- Centralized models: `models/centralized/`
- Federated models: `models/federated/`
- DDQN model: `models/rlhf/dqn_agent.pt`
