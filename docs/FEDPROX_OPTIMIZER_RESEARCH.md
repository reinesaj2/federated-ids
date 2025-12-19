# FedProx Optimizer Implementation Research

**Date:** 2025-12-15
**Context:** Research conducted to verify FedProx optimizer implementation against Li et al. (2020)
**Related PR:** #181

## Executive Summary

This document presents comprehensive research into FedProx optimizer implementations across the original paper, reference implementations, and popular federated learning frameworks. The research reveals critical findings about optimizer selection and identifies a fundamental issue with PR #181's approach.

**Critical Finding:** No reference implementation switches optimizers based on the mu (proximal term) value. The proximal term is optimizer-agnostic and works with any gradient-based optimizer (SGD, Adam, AdamW).

## Research Methodology

Three parallel research agents conducted comprehensive searches:

1. **Agent 1:** Analyzed the original Li et al. (2020) paper from MLSys 2020
2. **Agent 2:** Examined reference implementations (TensorFlow and PyTorch)
3. **Agent 3:** Surveyed community implementations across popular frameworks

## Findings from Original Paper (Li et al. 2020)

### Optimizer Specification

**Direct Quote:**
> "In order to draw a fair comparison with FedAvg, we employ SGD as a local solver for FedProx"

**Key Points:**
- SGD is used for BOTH FedAvg (mu=0) and FedProx (mu>0)
- Reason stated: "fair comparison" - NOT technical incompatibility
- Paper emphasizes flexibility: "Device k uses its local solver of choice to approximately minimize the objective"

### Hyperparameters

**Learning Rates (dataset-specific):**
- MNIST: 0.03
- FEMNIST: 0.003
- Shakespeare: 0.8
- Sent140: 0.3

**Proximal Term (mu):**
- Tested values: {0.001, 0.01, 0.1, 1}
- Best mu values vary by dataset: 1, 1, 1, 0.001, and 0.01

**Adaptive mu Strategy:**
- "Increase mu by 0.1 whenever the loss increases and decrease it by 0.1 whenever the loss decreases for 5 consecutive rounds"
- Initialize to mu=1 for IID data, mu=0 for non-IID datasets

**Note:** Full hyperparameter tables are in Appendix C.2 of the paper.

### Theoretical Foundation

The FedProx local subproblem:
```
h_k(w; w^t) = F_k(w) + (mu/2)||w - w^t||^2
```

Where the proximal term `(mu/2)||w - w^t||^2` prevents large parameter deviations from the global model.

**Critical Insight:** The proximal term is a regularization term added to the loss function. It does not fundamentally require a specific optimizer - it works with any gradient-based optimization method.

## Reference Implementation Analysis

### 1. Original Implementation (litian96/FedProx - TensorFlow)

**Repository:** https://github.com/litian96/FedProx

**Optimizer Implementation:**
- File: `/flearn/optimizer/pgd.py` (PerturbedGradientDescent)
- Type: Custom TensorFlow optimizer
- Update rule: `var_update = state_ops.assign_sub(var, lr_t*(grad + mu_t*(var-vstar)))`

**Configuration:**
- Learning Rate: 0.01 (in run scripts)
- Mu: 0 (must be set via command line)
- Momentum: NOT USED (custom optimizer)
- Weight Decay: NOT EXPLICITLY CONFIGURED

**Key Finding:** Same optimizer used for all mu values (0 and >0).

### 2. FedNova Implementation (JYWa/FedNova - PyTorch)

**Repository:** https://github.com/JYWa/FedNova

**Optimizer Implementation:**
- File: `/distoptim/FedProx.py`
- Type: Custom PyTorch optimizer extending `torch.optim.Optimizer`
- Proximal term added to gradient: `d_p.add_(self.mu, p.data - param_state['old_init'])`

**Default Parameters:**
```python
optimizer = FedProx(model.parameters(),
                    lr=0.1,
                    momentum=0.0,      # No momentum
                    mu=0,              # Set for FedProx
                    weight_decay=1e-4,
                    nesterov=False)
```

**Key Finding:** Same optimizer for all mu values. Momentum=0.0 by default.

### 3. PyTorch Implementation (ki-ljl/FedProx-PyTorch)

**Repository:** https://github.com/ki-ljl/FedProx-PyTorch

**Optimizer Implementation:**
- File: `client.py`
- Type: Standard PyTorch optimizers (Adam OR SGD)
- Proximal term added to loss:
```python
proximal_term = 0.0
for w, w_t in zip(model.parameters(), global_model.parameters()):
    proximal_term += (w - w_t).norm(2)
loss = loss_function(y_pred, label) + (args.mu / 2) * proximal_term
```

**Configurations:**

For Adam:
```python
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=1e-4)
```

For SGD:
```python
optimizer = torch.optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=0.9,
                            weight_decay=1e-4)
```

**Key Finding:** Supports BOTH Adam and SGD. Same optimizer used regardless of mu value. User chooses optimizer via config, not automatic switching.

### 4. PyTorch Implementation (c-gabri/Federated-Learning-PyTorch)

**Repository:** https://github.com/c-gabri/Federated-Learning-PyTorch

**Default Configuration:**
- Optimizer: 'sgd'
- Arguments: 'lr=0.01,momentum=0,weight_decay=4e-4'
- Mu: 0 (set via --mu flag for FedProx)

**Key Finding:** Momentum=0 explicitly. Same optimizer for all mu values.

## Community Implementations Survey

### Popular Framework Implementations

#### Flower Framework
- Uses "SGD with proximal term" per baseline documentation
- Focuses on proximal term modification rather than prescribing specific optimizer
- Strategy allows flexible optimizer configuration

#### TensorFlow Federated
- API: `build_weighted_fed_prox`
- Allows flexible `client_optimizer_fn` parameter
- Default server optimizer: SGD with learning rate 1.0

### Research on Optimizer Performance

#### Empirical Studies

**Adaptive Optimizers Can Outperform SGD:**
- Research shows "FedProx+Adam achieves the highest test accuracy and faster convergence speed" on FEMNIST dataset
- Adaptive optimizers (Adam, Adagrad, Yogi) can "significantly improve the performance of federated learning"
- For Transformer models: "Local SGD is still significantly worse than Local AdamW"

**No Universal Winner:**
- "The answers to which optimizer performs best highly vary depending on the setting"
- "No single algorithm works best across different performance metrics"
- SGD requires extensive hyperparameter tuning (grid-searching 11-13 step sizes)

### Known Implementation Issues

#### A. Momentum Buffer Management
- "Less accurate gradients at the end of local training count for more in the cumulative momentum, making the aggregated momentum biased and suboptimal"
- For Adam/RMSProp: "There exists an implicit momentum bias in the preconditioner"
- Server-side aggregation of optimizer states adds complexity

#### B. Proximal Term Bugs
- Flower PR #1513 fixed bug where "The FedProx proximal term always evaluated to 0"
- PyTorch forum discussions reveal implementations missing the mu coefficient
- GitHub Issue #6 on official repo showed confusion about proximal term implementation

#### C. Optimizer State Management
- PySyft Issue #3507: "Adam and other stateful optimizers do not work on federated models"
- Correction techniques are "hard to fundamentally solve for stochastic optimizers, such as SGDM, Adam, AdaGrad"

## Summary of Reference Implementations

| Implementation | Optimizer | Momentum | Weight Decay | Switches on mu? |
|---|---|---|---|---|
| Original (TensorFlow) | Custom SGD | N/A | N/A | No |
| FedNova (PyTorch) | Custom SGD | 0.0 | 1e-4 | No |
| ki-ljl (PyTorch) | Adam OR SGD | 0.9 (SGD) / N/A (Adam) | 1e-4 | No |
| c-gabri (PyTorch) | SGD | 0.0 | 4e-4 | No |
| Flower Framework | Configurable | Configurable | Configurable | No |
| TensorFlow Federated | Configurable | Configurable | Configurable | No |

**Universal Finding:** ZERO implementations switch optimizers based on mu value.

## Analysis of PR #181 Approach

### Current Implementation

```python
FEDPROX_MOMENTUM = 0.0

def _create_optimizer(parameters, lr, weight_decay, fedprox_mu):
    if fedprox_mu > 0.0:
        # FedProx uses SGD per Li et al. (MLSys 2020); no weight decay
        return torch.optim.SGD(parameters, lr=lr,
                              momentum=FEDPROX_MOMENTUM,
                              weight_decay=0.0)
    return create_adamw_optimizer(parameters, lr=lr,
                                  weight_decay=weight_decay)
```

### Issues Identified

#### 1. Automatic Optimizer Switching
**Problem:** Switches from AdamW to SGD when mu > 0.

**Why This Is Wrong:**
- No reference implementation does this
- Breaks hyperparameter consistency within experiments
- Not supported by Li et al. paper (they use SGD for BOTH mu=0 and mu>0)
- Contradicts research showing FedProx+Adam can outperform FedProx+SGD

#### 2. Weight Decay Zeroing
**Problem:** Sets `weight_decay=0.0` when mu > 0.

**Why This Is Wrong:**
- Most reference implementations use weight_decay=1e-4 for both FedAvg and FedProx
- Not specified in Li et al. paper
- Changes regularization behavior unexpectedly

#### 3. Codebase Inconsistency
**Problem:** Rest of codebase is tuned for AdamW.

**Impact:**
- Existing experiments used AdamW with tuned learning rates
- Switching to SGD invalidates prior hyperparameter tuning
- Would require re-tuning all learning rates for fair comparison

#### 4. Misinterpretation of Paper
**Problem:** Comment states "FedProx uses SGD per Li et al."

**Correction:**
- Paper says: "To draw a fair comparison with FedAvg, we employ SGD"
- This means: use SGD for BOTH FedAvg and FedProx (consistent optimizer)
- Does NOT mean: switch to SGD only when mu > 0

## Theoretical Justification

### Why Proximal Term Works with Any Optimizer

The FedProx objective:
```
min_w { F_k(w) + (mu/2)||w - w^t||^2 }
```

Gradient:
```
∇[F_k(w) + (mu/2)||w - w^t||^2] = ∇F_k(w) + mu(w - w^t)
```

**Key Insight:** The proximal term contributes an additive term to the gradient. Any gradient-based optimizer can process this modified gradient.

**Implementation Approaches:**
1. **Add to loss** (most common): `loss = F_k(w) + (mu/2)||w - w^t||^2`
2. **Add to gradient** (custom optimizer): `grad = ∇F_k(w) + mu(w - w^t)`

Both approaches work with any base optimizer (SGD, Adam, AdamW, etc.).

### When Adam/AdamW May Be Preferred

**Advantages:**
- Adaptive learning rates per parameter
- Often faster convergence on complex architectures
- Less sensitive to learning rate tuning
- Better for heterogeneous data distributions

**Research Evidence:**
- "FedProx+Adam achieves highest test accuracy" on FEMNIST
- "Local AdamW" significantly better than "Local SGD" for Transformers

## Recommendations

### Option 1: Use AdamW for All Cases (RECOMMENDED)

**Rationale:**
- Maintains consistency with existing codebase
- Supported by research showing Adam can outperform SGD with FedProx
- Minimal code changes required
- No need to re-tune hyperparameters

**Implementation:**
```python
def train_epoch(..., fedprox_mu=0.0, ...):
    # Always use AdamW
    optimizer = create_adamw_optimizer(model.parameters(),
                                      lr=lr,
                                      weight_decay=weight_decay)

    # Add proximal term to loss when mu > 0
    if fedprox_mu > 0.0 and global_params is not None:
        proximal_term = 0.0
        for p, p_t in zip(model.parameters(), global_tensors):
            proximal_term += (p - p_t).pow(2).sum()
        loss = loss + (fedprox_mu / 2) * proximal_term
```

### Option 2: Use SGD for All Cases

**Rationale:**
- Exact match to Li et al. original paper
- Simpler optimizer state (no momentum buffers in federated setting)
- Potentially easier to analyze theoretically

**Challenges:**
- Requires re-tuning ALL learning rates
- May require momentum tuning (0.0 vs 0.9)
- Disruptive to existing experiments
- May reduce performance on complex architectures

**Implementation:**
```python
def train_epoch(..., fedprox_mu=0.0, ...):
    # Always use SGD
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=0.0,  # or 0.9
                                weight_decay=weight_decay)

    # Proximal term added to loss as before
```

### Option 3: Make Optimizer Configurable

**Rationale:**
- Maximum flexibility for experiments
- Can compare SGD vs AdamW performance
- Avoids hard-coding optimizer choice

**Implementation:**
```python
def _create_optimizer(parameters, lr, weight_decay, optimizer_type='adamw'):
    if optimizer_type == 'sgd':
        return torch.optim.SGD(parameters, lr=lr,
                              momentum=0.0,
                              weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr,
                                weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
```

**Configuration:**
Add to runtime config: `"optimizer": "adamw"` or `"optimizer": "sgd"`

## Conclusion

The research definitively shows that:

1. **The proximal term is optimizer-agnostic** - it works with any gradient-based optimizer
2. **No reference implementation switches optimizers based on mu** - the choice of optimizer is independent of whether FedProx is enabled
3. **Both SGD and Adam/AdamW are valid choices** - research shows task-dependent performance
4. **Li et al. used SGD for consistency** - not because of technical requirements

**PR #181's approach of automatically switching optimizers is not supported by any reference implementation or theoretical justification.**

**Recommended Action:** Revise PR to use AdamW for all cases (Option 1) to maintain codebase consistency and align with research showing competitive or superior performance of adaptive optimizers with FedProx.

## References

### Primary Sources
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. Proceedings of Machine Learning and Systems, 2, 429-450.
- FedProx Paper (arXiv): https://arxiv.org/abs/1812.06127
- FedProx Paper (MLSys 2020): https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf

### Reference Implementations
- Official FedProx (TensorFlow): https://github.com/litian96/FedProx
- FedNova (PyTorch): https://github.com/JYWa/FedNova
- ki-ljl FedProx-PyTorch: https://github.com/ki-ljl/FedProx-PyTorch
- c-gabri Federated-Learning-PyTorch: https://github.com/c-gabri/Federated-Learning-PyTorch

### Framework Implementations
- Flower FedProx Baseline: https://flower.ai/docs/baselines/fedprox.html
- Flower FedProx Strategy: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedProx.html
- TensorFlow Federated: https://www.tensorflow.org/federated/api_docs/python/tff/learning/algorithms/build_weighted_fed_prox

### Research Papers
- Reddi, S., Charles, Z., Zaheer, M., et al. (2021). Adaptive federated optimization. ICLR 2021. https://arxiv.org/pdf/2003.00295
- Wang, J., et al. (2024). An empirical study of efficiency and privacy of federated learning algorithms. https://arxiv.org/html/2312.15375v1
- Marfoq, O., et al. (2024). Not all federated learning algorithms are created equal. https://arxiv.org/html/2403.17287v1

### GitHub Issues and Discussions
- FedProx Issue #6: Proximal term implementation: https://github.com/litian96/FedProx/issues/6
- Flower PR #1513: FedProx MNIST baseline: https://github.com/adap/flower/pull/1513
- PyTorch Forum: FedProx loss implementation: https://discuss.pytorch.org/t/help-fedprox-loss-implementation/93375
- PySyft Issue #3507: Stateful optimizers: https://github.com/OpenMined/PySyft/issues/3507
