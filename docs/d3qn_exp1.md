# D3QN Experiment 1: Double Dueling DQN + Prioritized Experience Replay

## Overview

This experiment implements **D3QN** — Double Dueling Deep Q-Network with Prioritized Experience
Replay — for the Surround Atari game. It builds on the class-map CNN approach proven in `exp16`
and adds three targeted algorithmic improvements to address specific failure modes observed across
all previous runs.

---

## Diagnosis of Previous Experiments

| Experiment | Mean Steps | Mean Reward | Notes |
|---|---|---|---|
| baseline_grayscale | 83.5 | -0.99 | Raw pixels; Q diverges |
| baseline_state | 168.2 | -0.711 | MLP on 7-tuple; limited capacity |
| dqn_long_exp11 | **254.7** | **-0.304** | Best ever; stopped at 7 K eps |
| exp16 (latest) | 209.2 | -0.623 | Best complete run; 32 K eps |
| PPO (all) | ~64 | ~-1.0 | Failed entirely with state-tuple |

Three structural problems limit the vanilla DQN (exp16):

1. **Q-value overestimation** — the standard DQN target `r + γ·max_a Q_target(s',a)` is
   positively biased because it uses the same network to select *and* evaluate the next action.
   This causes Q-values to drift upward and eventually produce a policy that chases phantom value.
   In exp16, `mean_q` oscillates wildly from −4 to +14.

2. **Undifferentiated value estimation** — in Surround, many states have a similar intrinsic
   danger level independent of which direction you move. A standard linear output head must learn
   this from scratch for every action. A **dueling architecture** learns `V(s)` (how good is this
   state?) and `A(s,a)` (how much better is action a than average?) independently, yielding
   better generalisation to unvisited states.

3. **Uniform random replay is wasteful** — with a 10 K buffer and uniform sampling, the agent
   revisits mundane mid-game transitions far more than the rare, high-surprise transitions near
   walls and dead-ends that actually carry gradient signal. A **prioritized replay** fixes this by
   sampling proportional to |TD error|^α, focusing training effort where it matters.

Additional factors:
- The 10 K replay buffer is too small (covers only ~50 episodes at mean 209 steps). A 100 K
  buffer provides more diverse experience and slower forgetting.
- Epsilon decays over only 1 % of training episodes (~500 episodes). Extending the decay window
  to 15 % (7 500 episodes) ensures genuine exploration throughout early training.

---

## Algorithm: D3QN + PER

### 1. Double DQN (DDQN)

Instead of:
```
target = r + γ · max_a Q_target(s', a)          # standard DQN (overestimates)
```
use:
```
a* = argmax_a Q_policy(s', a)                   # policy net picks action
target = r + γ · Q_target(s', a*)               # target net evaluates it
```
This decouples action *selection* from action *evaluation*, removing the positive bias without
introducing another network.

### 2. Dueling Network Architecture

The convolutional backbone is unchanged (3 × Conv2d → ReLU). The fully connected head is split:

```
feat ──► val_fc1 (256) ──► val_fc2 (1)    = V(s)
     └─► adv_fc1 (256) ──► adv_fc2 (4)   = A(s,a)

Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)         (Wang et al. 2016)
```

Subtracting the mean advantage ensures identifiability (V and A cannot be shifted by a
constant without changing Q) and speeds convergence.

### 3. Prioritized Experience Replay (PER)

A **SumTree** (binary segment tree) stores transition priorities so that:
- Sampling a mini-batch is **O(log N)** regardless of buffer size.
- Priority `p_i = (|δ_i| + ε)^α` where `δ_i` is the TD error, `ε = 1e-5`, `α = 0.6`.
- **Importance sampling weights** `w_i = (N · p_i / Σp)^{−β}` correct the introduced bias.
  `β` anneals from 0.4 → 1.0 over 50 K episodes so IS correction is mild early on when
  variance is high, then exact by the end.
- After each gradient step, priorities are updated with the fresh TD errors.

### 4. Supporting Improvements

| Hyperparameter | exp16 | D3QN exp1 | Rationale |
|---|---|---|---|
| Replay buffer | 10 K | 100 K | Less forgetting, more diverse batches |
| Epsilon decay | 1 % of eps (500) | 15 % of eps (7 500) | More exploration |
| Eps end | 0.01 | 0.05 | Prevent premature exploitation |
| Grad clip (norm) | ∞ | 1.0 | Conservative; PER already focuses gradients |
| State type | class_map | class_map | Proven best representation |
| Episodes | 50 K | 50 K | Same training budget |

---

## Implementation

All new code lives in `surround/dqn/train_d3qn.py`. No existing files are changed except
`surround/conf/constants.py` where `D3QN_*` constants are appended.

**Entry point:**
```bash
python -m surround.dqn.train_d3qn
# or equivalently
python surround/dqn/train_d3qn.py
```

**TensorBoard logs:** `runs/surround/dqn/d3qn_exp1/`
(same scalar tags as all other experiments for direct comparison)

**Checkpoints:** `runs/surround/dqn/d3qn_exp1/checkpoints/`

---

## Why This Should Win

- **Double DQN** alone typically yields +10–20 % improvement in Atari benchmarks by removing
  the overestimation bias that causes erratic Q-values.
- **Dueling networks** excel in environments with many states where the action choice is
  irrelevant (open-board Surround) and critical at state boundaries (near walls).
- **PER** has shown 30–50 % sample efficiency gains on hard Atari games; in Surround it should
  sharply reduce the number of episodes needed to learn wall-avoidance.
- Together, the 10× larger replay buffer and gentler epsilon schedule give the agent time to
  form a coherent policy before exploitation pressure builds.

The combination is well-validated (it is a subset of Rainbow DQN, Hessel et al. 2017) and
directly targets the failure modes observed in previous Surround runs.

---

## References

- van Hasselt et al. 2015 — "Deep Reinforcement Learning with Double Q-learning"
- Wang et al. 2016 — "Dueling Network Architectures for Deep Reinforcement Learning"
- Schaul et al. 2016 — "Prioritized Experience Replay"
- Hessel et al. 2017 — "Rainbow: Combining Improvements in Deep Reinforcement Learning"
