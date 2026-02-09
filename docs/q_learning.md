## Q-Learning State and Actions (Surround)

### State Vector (7 values)


See [this doc](https://docs.google.com/document/d/1oITOzKycnmKilUiKKwH4EgwXyeOtTVNaefD7BhEul1U/edit?tab=t.6lak97m9056e) for more context and results.

The Q-learning state is a 7-element tuple derived from the location extractor
and the definitions in [`surround/actions.py`](surround/actions.py):

1. D_UP (1 if ego is adjacent to a wall or out-of-bounds, else 0)
2. D_RIGHT (1 if ego is adjacent to a wall or out-of-bounds, else 0)
3. D_LEFT (1 if ego is adjacent to a wall or out-of-bounds, else 0)
4. D_DOWN (1 if ego is adjacent to a wall or out-of-bounds, else 0)
5. REL_X_OPP (0=opponent left, 1=same column, 2=opponent right)
6. REL_Y_OPP (0=opponent above, 1=same row, 2=opponent below)
7. last_action (1..4 for UP/RIGHT/LEFT/DOWN)

Total possible states with this encoding:

- `2^4 * 3^2 * 4 = 576`

### Action Choices (4 values)

The action space uses the minimal ALE action set for `ALE/Surround-v5` with
`full_action_space=False`. The action index order is:

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

