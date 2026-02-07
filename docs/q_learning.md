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

