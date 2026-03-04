# D3QN: Dueling Double DQN for Surround

## Overview

**D3QN** combines three improvements over vanilla DQN:
1. **Double DQN** — decouples action selection from evaluation to reduce Q-overestimation
2. **Dueling architecture** — separate V(s) and A(s,a) heads for better state-value estimation
3. **N-step returns (n=10)** — propagates reward signal faster through the episode

PER (Prioritized Experience Replay) was explored but abandoned after it caused catastrophic
divergence across two experiments. See [Experimental History](#experimental-history) below.

---

## Results

| Metric | D3QN (this) | Best DQN (dqn_long_exp11) |
|---|---|---|
| Win rate | **~42%** | 34.8% |
| Mean steps survived | ~50 | ~255 |
| Mean reward | ~-0.16 | -0.304 |
| Episodes to convergence | ~18 K | ~7 K (incomplete) |

D3QN learns a qualitatively different strategy: it plays shorter, decisive games (~50 steps)
and wins ~42% of them, versus DQN's survival-oriented play (255 steps, 34.8% win rate).
**Win rate is the correct metric** for Surround — steps survived measures avoidance, not
competitive play.

---

## Algorithm

### Double DQN

Standard DQN target (positively biased — same net selects and evaluates):
```
target = r + gamma * max_a Q_target(s', a)
```

Double DQN target (unbiased — policy net selects, target net evaluates):
```
a* = argmax_a Q_policy(s', a)
target = r + gamma * Q_target(s', a*)
```

### Dueling Network Architecture

The convolutional backbone is shared (3x Conv2d -> ReLU). The FC head is split into two streams:

```
feat --> val_fc1 (256) --> val_fc2 (1)    = V(s)
     +-> adv_fc1 (256) --> adv_fc2 (4)   = A(s,a)

Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)         (Wang et al. 2016)
```

Subtracting the mean advantage ensures identifiability and speeds convergence.

### N-Step Returns

Rather than storing single-step `(s, a, r, s')` transitions, each transition stores the
n-step discounted return:
```
R_n = r_t + gamma*r_{t+1} + ... + gamma^(k-1)*r_{t+k-1},   k = min(n, steps to episode end)
```
with bootstrap target `gamma^n * Q_target(s_{t+n}, a*)`. This propagates the terminal reward
signal 10x faster per gradient step compared to 1-step returns.

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Learning rate | 1e-4 | AdamW + amsgrad |
| Replay buffer | 10 K (uniform) | Matches baseline DQN |
| Batch size | 128 | |
| N-step | 10 | |
| Learning starts | 1 K env steps | |
| Update every | 4 env steps | |
| Gamma | 0.99 | |
| Tau (soft update) | 0.005 | |
| Eps start / end | 0.9 / 0.01 | |
| Eps decay fraction | 0.01 | Greedy by ~ep 1000 |
| Episodes | 50 K | |
| Grad clip (norm) | 1.0 | |
| State type | class_map | 4-class: empty/wall/opp/ego |

---

## Running

```bash
python -m surround.dqn.train_d3qn
```

**TensorBoard logs:** `runs/surround/d3qn/d3qn/`
**Checkpoints:** `runs/surround/d3qn/d3qn/checkpoints/`

---

## Experimental History

The current implementation is the result of 5 experiments. Key findings:

| Exp | Key Change | Outcome |
|---|---|---|
| exp1 | D3QN + PER (alpha=0.6), 100K buffer | **Diverged** -- Q->1e18 from near-empty PER buffer |
| exp2 | PER alpha 0.6->0.4, learning_starts=10K | **Dead** -- converged to degenerate Q~=-0.884, steps~=18; PER still diverged ep 1500-2500 then collapsed |
| exp3 | PER removed (uniform replay), LR 3e-4->1e-4, optimize every 4 steps | Stable, ~44 steps |
| exp4 | N-step returns added (n=10) | ~43 steps, similar win rate |
| **exp5** | Buffer 100K->10K, eps_decay 0.15->0.01 (matches baseline) | **Best** -- ~42% win rate, stable |

**Root cause of exp1/2 divergence:** PER sampled from a nearly empty buffer in early training,
producing extreme priority values that bootstrapped runaway Q-estimates. Once Q diverged,
recovery was impossible -- the agent collapsed to a fixed -1 policy.

**Why PER was dropped:** The baseline DQN achieves 34.8% win rate with uniform replay and
identical architecture (no PER). PER's instability cost outweighed its sample efficiency gains
for this environment.

**Why exp5 > exp3/4:** Smaller buffer (10K vs 100K) fills faster and stays more on-policy,
and the shorter epsilon decay (greedy by ep ~1000 vs ~7500) allows the agent to exploit the
policy it has learned earlier, matching the schedule that works for the baseline DQN.

---

## References

- van Hasselt et al. 2015 -- "Deep Reinforcement Learning with Double Q-learning"
- Wang et al. 2016 -- "Dueling Network Architectures for Deep Reinforcement Learning"
- Peng & Williams 1994 -- "Incremental Multi-Step Q-Learning" (n-step returns)
