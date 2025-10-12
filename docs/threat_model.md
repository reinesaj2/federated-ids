# Threat Model for Federated Intrusion Detection Systems

**Document Purpose:** This threat model defines the adversarial assumptions, attack scenarios, and defense mechanisms for the federated learning-based intrusion detection system (FL-IDS) implemented in this project.

**Scope:** Network-based intrusion detection using federated learning across multiple organizations with Byzantine-tolerant aggregation.

---

## 1. System Overview

### Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Central Server                               │
│  - Orchestrates FL rounds                                        │
│  - Aggregates model updates (FedAvg, Krum, Median, Bulyan)     │
│  - Distributes global model                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │ gRPC (Flower protocol)
          ┌──────────────┼──────────────┬──────────────┐
          │              │              │              │
     ┌────▼────┐    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
     │Client 0 │    │Client 1 │   │Client 2 │   │Client N │
     │(Org A)  │    │(Org B)  │   │(Org C)  │   │Adversary│
     └─────────┘    └─────────┘   └─────────┘   └─────────┘
      Local IDS      Local IDS     Local IDS     Malicious
      data (UNSW,    data          data          participant
      CIC-IDS2017)
```

**Implementation:** `server.py`, `client.py`, `robust_aggregation.py`

### Trust Boundaries

1. **Client-to-Server Communication:** Untrusted channel (model updates transmitted via gRPC)
2. **Server:** Semi-trusted aggregator (honest-but-curious: follows protocol but may observe updates)
3. **Clients:** Partially trusted (some clients may be Byzantine/malicious)
4. **Local Data:** Fully trusted (each client's IDS data remains local and private)

---

## 2. Adversary Model

### Threat Assumptions

| Assumption                    | Description                                                                                  | Implementation Reference                                       |
| ----------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Byzantine Clients**         | Up to `f` clients can behave arbitrarily (send malicious updates, collude, or fail silently) | `robust_aggregation.py:50` (`_guess_f_byzantine`)              |
| **Honest Majority**           | At least `n - f` clients are honest, where `n` is total clients                              | Required for Krum/Bulyan convergence                           |
| **Honest-but-Curious Server** | Server follows aggregation protocol but may attempt to infer client data from updates        | Motivates differential privacy (`client.py` DP implementation) |
| **No Sybil Attacks**          | Client identities are authenticated; attackers cannot spawn unlimited clients                | Out of scope (assumes identity management layer)               |
| **Static Adversary**          | Compromised clients are fixed before training begins (not adaptive)                          | Conservative assumption for current implementation             |

### Byzantine Tolerance Bounds

**Default conservative estimate:**

```python
# robust_aggregation.py:50-54
def _guess_f_byzantine(n: int) -> int:
    # Tolerate up to floor((n-2)/2) - 1 malicious clients, min 0
    if n <= 4:
        return 0
    return max(0, (n - 2) // 2 - 1)
```

**Example:** For 10 clients, defaults to `f = 3` (tolerates up to 30% Byzantine clients).

**Override:** Experiments can specify explicit `byzantine_f` parameter for tighter/looser bounds.

---

## 3. Attack Scenarios

### 3.1 Model Poisoning Attacks

**Objective:** Degrade global IDS model performance to increase false negatives (allow attacks through) or false positives (DoS via alert fatigue).

#### Attack Type: Gradient Ascent

**Description:** Adversarial clients invert their gradients to maximize loss instead of minimizing it.

**Implementation:**

```python
# client.py:329-331
mode = str(self.runtime_config.get("adversary_mode", "none"))
if mode == "grad_ascent":
    # Inverts gradients: model update pushed away from optimal direction
```

**Experiment Configuration:**

```python
# scripts/comparative_analysis.py:360
adversary_mode = "grad_ascent" if client_id < num_adversaries else "none"
```

**Impact:** FedAvg vulnerable (simple mean accepts all updates); robust methods mitigate via outlier detection.

#### Attack Type: Label Flipping (Future)

**Description:** Adversarial clients flip labels in their local training data (BENIGN ↔ ATTACK).

**Implementation:** `client.py:904` (choice available: `label_flip`, currently unused in comparative experiments)

**Impact:** More stealthy than gradient ascent; requires data-level poisoning detection.

### 3.2 Data Poisoning

**Objective:** Inject mislabeled or crafted samples into local training data to bias the global model.

**Status:** Not directly simulated; implicitly covered by label flipping mode.

**Real-world scenario:** Compromised organization injects fake benign traffic labeled as attacks to train false positive bias.

### 3.3 Inference Attacks

**Objective:** Infer sensitive information about other clients' IDS data from observed model updates.

**Threat Variants:**

- **Membership Inference:** Determine if a specific network flow was in a client's training set
- **Property Inference:** Learn aggregate properties (e.g., "Client X sees mostly DDoS attacks")
- **Model Inversion:** Reconstruct training samples from gradients

**Defense:** Differential privacy (client-side gradient clipping + noise addition)

**Implementation Status:**

- DP scaffold implemented (`client.py` with `--dp_enabled` flag)
- Privacy accounting not yet integrated (see `README.md:477-480`)

---

## 4. Defense Mechanisms

### 4.1 Robust Aggregation

**Purpose:** Tolerate Byzantine clients by filtering or down-weighting malicious updates based on geometric distance or statistical properties.

#### Method 1: Krum (`robust_aggregation.py:57-82`)

**Algorithm:** Select the single client update closest to others (minimum sum of squared distances to nearest neighbors).

**Byzantine Tolerance:** Provably robust if `f < n/3` (where `n` = total clients, `f` = malicious clients).

**Implementation:**

```python
# Krum selects 1 candidate based on proximity to n - f - 2 neighbors
selected = _krum_candidate_indices(vectors, f=f, multi=False)
return [arr.copy() for arr in weights_per_client[selected[0]]]
```

**Trade-off:** May discard honest updates that are legitimately different (e.g., due to non-IID data).

#### Method 2: Coordinate-wise Median (`robust_aggregation.py:96-102`)

**Algorithm:** Compute median of each model weight across all clients (layer-wise).

**Byzantine Tolerance:** Robust if `f < n/2` (50% Byzantine fraction).

**Implementation:**

```python
def _median_aggregate(weights_per_client: List[List[np.ndarray]]) -> List[np.ndarray]:
    for layer_idx in range(num_layers):
        stacked = _stack_layers(weights_per_client, layer_idx)
        aggregated.append(np.median(stacked, axis=0))
    return aggregated
```

**Trade-off:** High communication cost (all client updates needed); slower convergence on non-IID data.

#### Method 3: Bulyan (Simplified) (`robust_aggregation.py:105-124`)

**Algorithm:** Multi-Krum selection (select top `k` candidates) + coordinate-wise median over selected set.

**Byzantine Tolerance:** Stronger than Krum alone; tolerates up to `f < n/4` in theory (simplified version may differ).

**Implementation:**

```python
# Select k = n - f - 2 best candidates via Multi-Krum
selected = _krum_candidate_indices(vectors, f=f, multi=True)
# Then apply median aggregation over selected subset
```

**Trade-off:** Computationally expensive (O(n²) pairwise distance computation).

### 4.2 Differential Privacy (DP)

**Purpose:** Prevent inference attacks by adding calibrated noise to client updates, ensuring that no single training sample significantly affects the model.

**Mechanism:** Client-side gradient clipping + Gaussian noise

**Implementation Status:**

- Clipping and noise addition implemented (`client.py`)
- Privacy budget (ε, δ) tracking **not yet implemented** (acknowledged in `README.md:477-480`)

**Configuration:**

```bash
python client.py --dp_enabled --dp_noise_multiplier 1.0 --dp_clip_norm 1.0
```

**Trade-off:** Noise degrades model accuracy; privacy-utility curve explored in thesis Objective 4.

**Future Work:** Integrate privacy accountant (e.g., Opacus, TensorFlow Privacy) to track cumulative ε over FL rounds.

### 4.3 Secure Aggregation (SecAgg)

**Purpose:** Cryptographically mask individual client updates so the server only sees the aggregate, preventing honest-but-curious server from inspecting raw updates.

**Implementation Status:** **Stub only** (toggle logged but updates not encrypted).

**Acknowledged Limitation:**

```markdown
# README.md:477-480

- Secure Aggregation (stub): toggle provided and status logged, but updates are
  not cryptographically masked. Integration of secure summation/masking is
  planned for a later milestone.
```

**Planned Approach:** Use additive secret sharing or Paillier encryption (follow Bonawitz et al. 2017 protocol).

**Trade-off:** High communication/computation overhead; requires multiple rounds of key agreement.

---

## 5. Security Guarantees & Limitations

### Proven Guarantees

| Mechanism  | Guarantee                     | Condition                                    | Reference                      |
| ---------- | ----------------------------- | -------------------------------------------- | ------------------------------ |
| **Krum**   | Converges to honest mean      | `f < n/3` Byzantine clients                  | Blanchard et al., NeurIPS 2017 |
| **Median** | Tolerates up to 50% malicious | `f < n/2` Byzantine clients                  | Yin et al., ICML 2018          |
| **Bulyan** | Stronger Byzantine resistance | `f < n/4` (theoretical bound)                | Mhamdi et al., ICML 2018       |
| **DP**     | ε-differential privacy        | With proper accountant (not yet implemented) | McMahan et al., ICLR 2018      |

### Known Limitations

1. **Static Adversary Assumption:** Adaptive attackers that change strategy mid-training are not modeled.

2. **No Sybil Defense:** If attacker controls client identity registration, they can exceed `f` by spawning fake clients.

3. **Non-IID Amplifies Risk:** Legitimate heterogeneous updates may be misclassified as adversarial by robust aggregators (false positive rate increases).

4. **DP Accounting Gap:** Current implementation lacks cumulative privacy budget tracking; cannot certify (ε, δ)-DP guarantees.

5. **SecAgg Stub:** Server currently observes all raw updates; no cryptographic masking.

6. **Collusion Not Addressed:** If Byzantine clients collude and coordinate attacks, they may evade distance-based detection (Krum/Bulyan).

### Threat Model Boundaries (Out of Scope)

- **Physical Attacks:** Hardware tampering, side-channel attacks on client devices
- **Supply Chain Attacks:** Compromised dependencies in Flower/PyTorch libraries
- **Network-Level Attacks:** DDoS on FL server, man-in-the-middle (assumes TLS/secure transport)
- **Model Extraction:** Adversary queries deployed IDS model to reverse-engineer it (post-training threat)

---

## 6. Experimental Validation

### Attack Resilience Experiments

**Dimension:** Vary adversary fraction `{0%, 10%, 30%}` with FedAvg vs. robust methods.

**Configuration:** `scripts/comparative_analysis.py:120-139` (attack dimension)

**Metrics:**

- Macro-F1 score degradation vs. adversary percentage
- Convergence rate (rounds to reach target accuracy)
- Per-client variance (measure of fairness)

**Expected Results:**

- FedAvg: Significant accuracy drop at 30% adversaries
- Krum/Median/Bulyan: Maintained performance (within 5% of benign baseline)

**Visualization:** `scripts/generate_thesis_plots.py:695` (`plot_attack_resilience`)

### Non-IID Heterogeneity

**Challenge:** Non-IID data distributions (Dirichlet α=0.1) can cause robust aggregators to reject honest updates.

**Mitigation:** FedProx regularization (`--fedprox_mu 0.01`) reduces local drift, improving compatibility with robust methods.

**Experiment:** `scripts/compare_fedprox_fedavg.sh` (alpha grid sweep)

---

## 7. Threat Model Evolution

### Current State (Deliverable 2)

- [DONE] Byzantine-tolerant aggregation (Krum, Median, Bulyan)
- [DONE] Adversary simulation (gradient ascent mode)
- [PARTIAL] DP scaffold (no privacy accounting)
- [STUB] SecAgg (stub only)

### Planned Enhancements (Future Milestones)

1. **Privacy Accountant Integration:** Track cumulative ε across FL rounds using Opacus or TF Privacy.
2. **Secure Aggregation Protocol:** Implement cryptographic masking (Bonawitz et al. protocol).
3. **Adaptive Adversary Simulation:** Model attackers that adjust strategy based on observed global model changes.
4. **Anomaly-Based Detection:** Server-side Byzantine client detection (flag suspicious clients before aggregation).
5. **Reputation Systems:** Weight client contributions based on historical accuracy (discount previously malicious clients).

---

## 8. References

**Robust Aggregation:**

- Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent," NeurIPS 2017
- Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates," ICML 2018
- Mhamdi et al., "The Hidden Vulnerability of Distributed Learning in Byzantium," ICML 2018

**Privacy:**

- McMahan et al., "Learning Differentially Private Recurrent Language Models," ICLR 2018
- Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning," CCS 2017

**FL for IDS:**

- Li et al., "Federated Optimization in Heterogeneous Networks," MLSys 2020 (FedProx)
- Thesis proposal (`FL.txt`) - Objectives 1-5 covering robustness, non-IID handling, privacy, and personalization

---

## 9. Summary

This FL-IDS system operates under a **partial trust** model where:

- **Clients** are Byzantine-tolerant (up to `f < n/3` for Krum)
- **Server** is honest-but-curious (DP mitigates inference risk)
- **Data** remains local (privacy-preserving by design)

**Key takeaway:** Robust aggregation (Krum, Median, Bulyan) is the **primary defense** against model poisoning, while DP and SecAgg provide **secondary defenses** against inference attacks and server-side observation (DP scaffold exists, SecAgg planned).

**For thesis defense:** This threat model justifies design choices in all 5 objectives—robustness (Obj 1-2), heterogeneity (Obj 2), personalization (Obj 3), privacy (Obj 4), and evaluation (Obj 5).
