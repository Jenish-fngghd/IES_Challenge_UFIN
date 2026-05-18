# Model Artifacts

This folder stores trained model files and fitted scalers. Result summaries, plots, round histories, and metric JSON files remain under `results/`.

All paths below are relative to the repository root, `IES_Challenge_UFIN/`.

## Structure

```text
models/
├── centralized/
│   ├── tcn/                   # TCN weights and scaler
│   ├── gat/                   # GAT encoder and classifier weights
│   ├── catgnn/                # CaT-GNN encoder and classifier weights
│   ├── dae/                   # DAE model and scaler
│   └── fusion/                # Gated Fusion and XGBoost meta-learner artifacts
├── federated/
│   ├── fedavg_fedbn_best/   # canonical best federated model
│   ├── local_only/          # local-only client baseline models
│   └── legacy_algorithms/   # older FedAvg/FedNova/FedProx artifacts
└── rlhf/
    └── dqn_agent.pt         # DDQN threshold-control policy
```

## Canonical Federated Model

The final paper uses:

```text
models/federated/fedavg_fedbn_best/best_pipeline.pt
```

This corresponds to the FedAvg+FedBN run with best AUC 0.9290 at round 146. The duplicate file in the same folder was preserved from the original notebook output because it has the same hash as the canonical model.

## GitHub Note

Model files are ignored by `.gitignore` because they are large binary artifacts. If the repository is released publicly and these files need to be shared, use Git LFS or provide an external artifact link.
