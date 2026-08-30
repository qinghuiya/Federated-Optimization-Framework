# Algorithms and assumptions

Let `w_t` be the global model at round `t`, `w_i` a selected client's endpoint, and
`p_i = n_i / sum(n_j)` its sample weight. Floating-point model state is aggregated;
non-floating buffers are copied from the largest-weight client unless an algorithm says
otherwise.

## Basic methods

### FedSGD

Each selected client performs one mini-batch update in this simulation, followed by
sample-weighted averaging. This makes the communication pattern explicit, although a
full-gradient FedSGD experiment should set each client batch to its entire local dataset.

### FedAvg

The server sets `w_(t+1) = sum_i p_i w_i`. Any registered local optimizer may be used.
The canonical source is McMahan et al., [Communication-Efficient Learning of Deep
Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html),
AISTATS 2017.

### FedAvgM

The server computes the average client displacement `d_t = sum_i p_i(w_i - w_t)`, then
updates `v_t = beta v_(t-1) + d_t` and `w_(t+1) = w_t + eta_s v_t`. Configure `momentum`
and `server_lr` under `federated.server`.

## Client-drift methods

### FedProx

The local objective adds `mu/2 * ||w - w_t||^2`. Set `federated.proximal_mu`; aggregation
remains sample-weighted FedAvg. Source: Li et al., [Federated Optimization in Heterogeneous
Networks](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html),
MLSys 2020.

### SCAFFOLD

The local gradient is corrected by `c - c_i`. Client and global control variates persist
across rounds. This implementation follows the constant-step plain-SGD form and rejects
momentum or local schedules. Source: Karimireddy et al., [SCAFFOLD: Stochastic Controlled
Averaging for Federated Learning](https://proceedings.mlr.press/v119/karimireddy20a.html),
ICML 2020.

### FedNova

Each client displacement is normalized by its number of local SGD steps, the normalized
updates are averaged, and the result is rescaled by the sample-weighted average step
count. This implementation covers plain SGD without momentum; it rejects configurations
that would need the paper's more general effective-step coefficient. Source: Wang et al.,
[Tackling the Objective Inconsistency Problem in Heterogeneous Federated
Optimization](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html),
NeurIPS 2020.

## Adaptive server optimization (FedOpt)

The negative average client displacement is treated as a pseudo-gradient. FedAdagrad,
FedAdam, and FedYogi maintain server first/second-moment state. Configure `server_lr`,
`beta1`, `beta2`, and `tau`. Source: Reddi et al., [Adaptive Federated
Optimization](https://openreview.net/forum?id=LkFG3lB13U5), ICLR 2021.

## Robust aggregation

- **Coordinate Median** takes the median independently at every floating-point coordinate.
- **Trimmed Mean** removes the configured low/high fraction at every coordinate and
  averages the remainder.
- **Krum** selects the update with the smallest sum of distances to its nearest peers;
  it requires `n >= 2f + 3` participating clients.

These rules intentionally ignore sample weights because weighting changes their robust
estimators. Krum's source is Blanchard et al., [Machine Learning with Adversaries:
Byzantine Tolerant Gradient Descent](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html),
NeurIPS 2017. Median and trimmed-mean defenses are discussed by Yin et al., [Byzantine-Robust
Distributed Learning: Towards Optimal Statistical
Rates](https://proceedings.mlr.press/v80/yin18a.html), ICML 2018.

## Local optimizers

The local registry delegates to the corresponding `torch.optim` implementation:

| Name | Typical use |
|---|---|
| `sgd` | canonical FL baseline |
| `momentum`, `nesterov` | convenient SGD presets |
| `adam`, `adamw` | adaptive first-order training |
| `adagrad`, `adadelta`, `rmsprop` | classic adaptive methods |
| `asgd` | averaged SGD |
| `radam`, `nadam` | rectified/Nesterov Adam variants |
| `rprop` | resilient propagation; usually best with full-batch local gradients |

Options beneath `local.optimizer` are passed to PyTorch after the `name` field is removed.
Not every local optimizer is theoretically justified with every federated method. Treat
cross-products as experiments, not equivalences to a named paper.

