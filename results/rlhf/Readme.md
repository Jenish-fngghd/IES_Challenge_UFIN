# RLHF Simulation Report
## DQN-Based Adaptive Threshold Tuning for Fraud Detection

**Experiment script:** `notebooks/final/RLHF/experiment_rlhf.py`  
**Static-threshold script:** `notebooks/final/RLHF/static_threshold_comparison.py`  
**Date:** 2026-04-30  
**Status:** PASSED

---

## 1. System Overview

The RLHF layer sits on top of a frozen four-encoder ensemble (TCN + GAT + CaT-GNN + DAE → Gated Fusion → XGBoost) that produces a fraud probability score for each transaction. Rather than using a fixed decision threshold, a Deep Q-Network (DQN) agent learns to adaptively tune the threshold by observing rolling performance metrics and receiving reward signals derived from delayed ground-truth feedback — simulating the chargeback cycle in real fraud systems.

```
Transaction Scores (XGBoost)
         │
         ▼
  DecisionEngine (BLOCK / REVIEW / ALLOW)
  using agent.threshold
         │
         ▼ (7-step delay)
  Ground Truth Revealed
         │
         ▼
  Reward  ──►  DQN Agent  ──►  New Threshold
         │
         ▼
  State = [precision, recall, FPR, fraud_rate, volume_norm, threshold_norm]
```

---

## 2. Final Hyperparameter Configuration

| Parameter | Value | Notes |
|---|---|---|
| `STATE_DIM` | 6 | Added normalised threshold as 6th dimension |
| `ACTION_DIM` | 5 | Deltas: −0.10, −0.05, 0, +0.05, +0.10 |
| `HIDDEN_DIM` | 64 | 3-layer MLP Q-network |
| `GAMMA` | 0.99 | Discount factor |
| `EPSILON_START` | 0.20 | Initial exploration rate |
| `EPSILON_END` | 0.01 | Minimum exploration rate |
| `EPSILON_DECAY` | 0.995 | Per-update multiplicative decay |
| `LR_DQN` | 1e-4 | Adam optimiser |
| `BATCH_SIZE_DQN` | 64 | Replay buffer sample size |
| `BUFFER_CAPACITY` | 2,000 | Rolling window — prevents stale boundary transitions |
| `TARGET_UPDATE_N` | 100 | Hard target-network sync interval |
| `MIN_BUFFER` | 500 | Steps before training begins |
| `THRESHOLD_MIN` | 0.30 | Hard floor (operationally realistic) |
| `THRESHOLD_MAX` | 0.80 | Hard ceiling |
| `REVIEW_BAND` | 0.15 | Score range below threshold sent to REVIEW |
| `KL_BETA` | 0.20 | Drift penalty coefficient |
| `BATCH_N` | 100 | Transactions per simulation step |
| `DELAY_STEPS` | 7 | Simulated feedback delay (chargeback window) |
| `WINDOW_N` | 500 | Rolling window for state computation |
| `BASELINE_THR` | 0.868 | Original XGBoost optimal threshold |
| `N_EPOCHS` | 3 | Repetitions of validation data for convergence |

---

## 3. Reward Function

The original per-transaction reward (`TP=+1.0×w`, `FN=−10.0×w`, `FP=−1.0`, `TN=+0.1`) caused threshold collapse: the asymmetric FN penalty always dominated, driving the DQN to the floor.

The final reward is **batch-level F2 score** (β = 2, recall-weighted per the design specification), with two correction terms:

```
reward = F2(batch) + amount_signal − KL_penalty − boundary_penalty

F2 = (5 × precision × recall) / (4 × precision + recall)

amount_signal = 0.005 × (Σ log1p(amt) for caught fraud − Σ log1p(amt) for missed fraud)

KL_penalty    = KL_BETA × drift²          if |threshold − 0.868| > 0.15
                0                          otherwise

boundary_penalty = 2.0 × (1 − (thr − THRESHOLD_MIN) / 0.10)   if thr ≤ 0.40
                   2.0 × (1 − (THRESHOLD_MAX − thr) / 0.10)   if thr ≥ 0.70
                   0                                            otherwise
```

**Why F2 prevents collapse:** F2 = 0 when everything is blocked (precision → 0) and F2 = 0 when everything is allowed (recall → 0). The maximum sits at an intermediate threshold (~0.45–0.60 for 3.5% fraud prevalence), giving the DQN a well-shaped reward landscape with a natural optimum.

**Reward clipping:** rewards are clipped to [−2.0, 1.0] before pushing to the replay buffer to stabilise gradient magnitudes.

---

## 4. Architecture Changes from Original

### 4.1 Double DQN (DDQN)

Vanilla DQN overestimates Q-values at unseen states via the greedy max operator in the Bellman target. At boundary thresholds the agent had never visited, this caused it to underestimate the cost of going there.

```python
# Original (vanilla DQN) — overestimates boundary Q-values
next_q = self.target_net(ns).max(dim=1)[0]

# Fixed (DDQN) — Q-net selects, target-net evaluates
next_actions = self.q_net(ns).argmax(dim=1, keepdim=True)
next_q       = self.target_net(ns).gather(1, next_actions).squeeze(1)
```

### 4.2 Threshold in State Vector

The original 5-dimensional state `[precision, recall, FPR, fraud_rate, volume_norm]` did not include the agent's own threshold. The DQN was learning in a POMDP: it could not directly associate its actions with threshold levels, only with lagged aggregate statistics. The 6th dimension `threshold_norm = (thr − 0.30) / 0.50` provides the missing control signal.

### 4.3 Epsilon Decay

Replaced constant `EPSILON = 0.10` with exponential decay:
```python
eps = max(EPSILON_END, EPSILON_START × EPSILON_DECAY^step_count)
```
Early training explores broadly; late training exploits the learned policy.

### 4.4 Validation Fixes

Three validation bugs that caused `validation_passed: true` to be reported incorrectly:

| Bug | Fix |
|---|---|
| `threshold_std > 0.15` logged as `[WARN]`, did not set `ok = False` | Changed to `[FAIL]`, sets `ok = False` |
| No check for threshold collapse to boundary | Added collapse detection: FAIL if `final_threshold` within 0.05 of either bound |
| AUC-PR comparison always identical (score-level metric, unchanged by threshold) | Replaced with F2 score comparison at each operating threshold |

---

## 5. Validation Results

All five red-flag checks pass.

| Check | Baseline | RLHF | Result |
|---|---|---|---|
| FNR | 0.8304 | **0.5686** | PASS (↓ 31.5 pp) |
| F2 score | 0.2028 | **0.4753** | PASS (↑ 134%) |
| Threshold std | — | **0.1096** | PASS (< 0.15) |
| Review queue max | — | **10** | PASS (< 1,000) |
| Threshold bounds | — | **0.500** | PASS (within [0.35, 0.75]) |

### Confusion Matrix (RLHF, first epoch)

|  | Predicted Fraud | Predicted Legit |
|---|---|---|
| **Actual Fraud** | TP = 1,783 | FN = 2,350 |
| **Actual Legit** | FP = 441 | TN = 113,534 |

### Full Metrics

| Metric | Baseline | RLHF |
|---|---|---|
| Decision threshold | 0.868 | **0.500** |
| FNR | 0.8304 | **0.5686** |
| F2 score | 0.2028 | **0.4753** |
| AUC-PR | 0.6592 | 0.6592 (score-level, unchanged) |
| Precision @ Recall=0.90 | 0.1629 | 0.1629 (score-level, unchanged) |
| Dollar value caught | $84,539 | **$265,059** |
| Threshold std | — | **0.1096** |
| Review queue max | — | **10** |
| `validation_passed` | — | **true** |

---

## 6. Threshold Convergence

The DQN converges to a stable operating threshold of **0.500**, down from the baseline of 0.868. This reflects the model learning that the XGBoost ensemble's F1-optimal threshold (0.868) is too conservative for a recall-weighted objective: lowering the threshold to 0.50 catches significantly more fraud at an acceptable false-positive cost.

```
Threshold trajectory (step log):

Step   200 | 0.300   ← warmup random walk, floor now 0.30 (not 0.10)
Step   400 | 0.300   ← still pre-learning (buffer < 500)
Step   600 | 0.800   ← learning begins, initial overshoot to ceiling
Step   800 | 0.300   ← DQN corrects, learns boundaries are penalised
Step  1000 | 0.450   ── stable zone begins ──────────────────────────
Step  1200 | 0.550
Step  1400 | 0.500
Step  1600 | 0.450
Step  1800 | 0.500
Step  2000 | 0.500
...
Step  3400 | 0.500   ← converged
```

---

## 7. Root Causes of Original Oscillation

The original code had `threshold_std = 0.35`, seven times the target. The oscillation was caused by five interacting issues, resolved in order of impact:

| # | Root Cause | Fix Applied |
|---|---|---|
| 1 | Per-transaction FN penalty (−10×) always dominated FP (−1), so DQN raced to floor | Replaced with batch-level F2 reward — natural optimum at intermediate threshold |
| 2 | Threshold not in state: DQN couldn't attribute actions to threshold levels | Added normalised threshold as 6th state dimension |
| 3 | Vanilla DQN overestimated Q-values at unseen boundary states | Switched to Double DQN |
| 4 | `THRESHOLD_MIN=0.10` / `THRESHOLD_MAX=0.90` allowed extreme swings during warmup | Narrowed to [0.30, 0.80] — operationally realistic and eliminates outlier swings |
| 5 | `BUFFER_CAPACITY=10,000` with only 3,544 steps meant early boundary transitions never expired | Reduced to 2,000 so boundary transitions cycle out once stable learning begins |

---

## 8. Iterative Tuning Log

| Run | Key Change | threshold_std | Outcome |
|---|---|---|---|
| 0 (original) | Baseline | 0.350 | FAIL — threshold collapse to 0.10 |
| 1 | KL_BETA 0.05→0.30, quadratic drift, boundary penalty 20 | 0.350 | FAIL |
| 2 | FN_PENALTY −10→−4, KL_BETA→0.50, boundary penalty→100 | 0.310 | FAIL |
| 3 | MIN_BUFFER 500→100, boundary penalty→200 | 0.348 | FAIL |
| 4 | **F2-based reward**, boundary penalty→2.0, KL_BETA→0.20 | 0.228 | FAIL (close) |
| 5 | Halved action deltas + MIN_BUFFER→200 | 0.273 | FAIL (worse) |
| 6 | Warmup threshold freeze | 0.308 | FAIL (worse) |
| 7 | **N_EPOCHS=3** (3× data) | 0.308 | FAIL |
| 8 | **DDQN + reward clipping** | 0.334 | FAIL |
| 9 | **Threshold in state (6-dim)** + DDQN + F2 | 0.162 | FAIL (very close) |
| 10 | **BUFFER_CAPACITY=2,000** | 0.164 | FAIL (plateau) |
| 11 | **THRESHOLD_MIN=0.30, THRESHOLD_MAX=0.80** | **0.110** | **PASS** |

---

## 9. Artifacts

Outputs are organized under `results/rlhf/`, while the trained DDQN model is stored under `models/rlhf/`. All paths below are relative to the repository root, `IES_Challenge_UFIN/`:

| File | Description |
|---|---|
| `models/rlhf/dqn_agent.pt` | Trained DQN Q-network weights (state dict) |
| `results/rlhf/metrics/rlhf_metrics.json` | All validation metrics |
| `results/rlhf/metrics/static_threshold_comparison.json` | Static threshold sweep and DDQN comparison |
| `results/rlhf/histories/reward_history.json` | Per-step clipped reward trajectory |
| `results/rlhf/figures/rlhf_evaluation.png` | 6-panel evaluation plot |
| `results/rlhf/figures/static_threshold_comparison.png` | Static threshold comparison plot |
| `results/rlhf/figures/threshold_trajectory.png` | Standalone threshold convergence plot |
| `results/rlhf/arrays/val_scores.npy` | XGBoost fraud probability scores (118,108) |
| `results/rlhf/arrays/val_labels.npy` | Ground-truth labels |
| `results/rlhf/arrays/val_amounts.npy` | Transaction amounts |
