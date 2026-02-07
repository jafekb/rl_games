## Q-Learning State and Actions (Surround)

### State Vector (6 values)


See [this doc](https://docs.google.com/document/d/1oITOzKycnmKilUiKKwH4EgwXyeOtTVNaefD7BhEul1U/edit?tab=t.6lak97m9056e) for more context and results.

The Q-learning state is a 6-element tuple derived from the RAM extractor:

1. Left distance to nearest wall (non-negative)
2. Right distance to nearest wall (non-negative)
3. Up distance to nearest wall (non-negative)
4. Down distance to nearest wall (non-negative)
5. dx to opponent (opponent_x - self_x, signed)
6. dy to opponent (opponent_y - self_y, signed)

Clipping is applied to reduce state space size:

- Distances are clipped to `[0, 7]`
- dx/dy are clipped to `[-7, 7]`

Total possible states with this clipping:

- `8^4 * 15^2 = 921,600`

### Action Choices (5 values)

The action space uses the minimal ALE action set for `ALE/Surround-v5` with
`full_action_space=False`. The action index order is:

0. NOOP
1. UP
2. RIGHT
3. LEFT
4. DOWN

### Training Improvements (Post-First Run)

After the first benchmark run, we made targeted changes to improve learning
stability and coverage:

- Longer training: increased total episodes and cycles to capture more state
  transitions.
- Optimistic initialization: new states start with `np.ones` to encourage early
  exploration and faster differentiation between actions.
- Epsilon decay: exploration starts high and decays toward a minimum to shift
  from exploration to exploitation as training progresses.
- Reward shaping: a small per-step reward (+0.01) for non-terminal steps
  encourages survival and longer rollouts.
- Environment similarity: set the difficulty to 0 during training instead of 1 so that the opponent is the same during training and eval.

