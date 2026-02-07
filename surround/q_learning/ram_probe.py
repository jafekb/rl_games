from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import ale_py
import gymnasium as gym
import numpy as np

ACTION_IDS = {
    "NOOP": 0,
    "UP": 1,
    "RIGHT": 2,
    "LEFT": 3,
    "DOWN": 4,
}

PRE_TURN_STEPS = 3


@dataclass(frozen=True)
class SweepConfig:
    name: str
    action: int
    steps: int


@dataclass(frozen=True)
class CandidateStat:
    index: int
    corr: float
    trend: float
    value_min: int
    value_max: int


@dataclass(frozen=True)
class PositionIndices:
    self_x: int
    self_y: int
    opp_x: int
    opp_y: int


@dataclass(frozen=True)
class Bounds:
    x_min: int
    x_max: int
    y_min: int
    y_max: int


def make_env(difficulty: int, mode: int):
    gym.register_envs(ale_py)
    return gym.make(
        "ALE/Surround-v5",
        obs_type="ram",
        full_action_space=False,
        difficulty=difficulty,
        mode=mode,
        render_mode=None,
    )


def run_sweep(
    env,
    action: int,
    steps: int,
    seed: int,
    noop_every: int,
    pre_actions: Sequence[int] | None = None,
) -> np.ndarray:
    observation, _info = env.reset(seed=seed)
    if pre_actions:
        for pre_action in pre_actions:
            observation, _reward, terminated, truncated, _info = env.step(pre_action)
            if terminated or truncated:
                break
    frames = [observation.copy()]
    for _ in range(steps):
        observation, _reward, terminated, truncated, _info = env.step(action)
        frames.append(observation.copy())
        if terminated or truncated:
            break
        for _ in range(noop_every):
            observation, _reward, terminated, truncated, _info = env.step(ACTION_IDS["NOOP"])
            frames.append(observation.copy())
            if terminated or truncated:
                break
        if terminated or truncated:
            break
    return np.asarray(frames, dtype=np.uint8)


def make_multiagent_env(difficulty: int, mode: int, render_mode: str | None = None):
    try:
        from pettingzoo.atari import surround_v2
    except ImportError as exc:
        raise ImportError(
            "PettingZoo is required for multi-agent probing. Install pettingzoo to proceed."
        ) from exc

    rom_path = str(Path("~/.local/share/AutoROM/roms").expanduser())
    kwargs = {
        "obs_type": "ram",
        "full_action_space": False,
        "auto_rom_install_path": rom_path,
        "render_mode": render_mode,
    }

    def filter_kwargs(env_fn):
        try:
            import inspect

            params = inspect.signature(env_fn).parameters
            if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
                return kwargs
            return {key: value for key, value in kwargs.items() if key in params}
        except (TypeError, ValueError):
            return kwargs

    if hasattr(surround_v2, "parallel_env"):
        return surround_v2.parallel_env(**filter_kwargs(surround_v2.parallel_env))
    return surround_v2.env(**filter_kwargs(surround_v2.env))


def run_multiagent_sweep(
    env,
    moving_agent: str,
    action: int,
    steps: int,
    seed: int,
    fixed_action: int,
    noop_every: int,
    pre_actions: Sequence[int] | None = None,
) -> np.ndarray:
    frames: list[np.ndarray] = []
    reset_out = env.reset(seed=seed)
    observations = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    is_parallel = isinstance(observations, dict)
    if is_parallel:
        if moving_agent not in observations:
            raise ValueError(f"Unknown moving agent: {moving_agent}")
        if pre_actions:
            for pre_action in pre_actions:
                actions = {
                    agent: (pre_action if agent == moving_agent else fixed_action)
                    for agent in env.agents
                }
                observations, _rewards, terminations, truncations, _infos = env.step(actions)
                if terminations.get(moving_agent) or truncations.get(moving_agent):
                    break
        frames.append(observations[moving_agent].copy())
        for _ in range(steps):
            actions = {
                agent: (action if agent == moving_agent else fixed_action) for agent in env.agents
            }
            observations, _rewards, terminations, truncations, _infos = env.step(actions)
            if moving_agent in observations:
                frames.append(observations[moving_agent].copy())
            if terminations.get(moving_agent) or truncations.get(moving_agent):
                break
            for _ in range(noop_every):
                actions = dict.fromkeys(env.agents, fixed_action)
                observations, _rewards, terminations, truncations, _infos = env.step(actions)
                if moving_agent in observations:
                    frames.append(observations[moving_agent].copy())
                if terminations.get(moving_agent) or truncations.get(moving_agent):
                    break
            if terminations.get(moving_agent) or truncations.get(moving_agent):
                break
        return np.asarray(frames, dtype=np.uint8)

    if not hasattr(env, "agent_iter"):
        raise ValueError("Unsupported PettingZoo environment interface.")

    if moving_agent not in env.possible_agents:
        raise ValueError(f"Unknown moving agent: {moving_agent}")

    order = env.possible_agents
    if pre_actions:
        for pre_action in pre_actions:
            for agent in env.agent_iter():
                observation, _reward, terminated, truncated, _info = env.last()
                action_to_take = (
                    None
                    if (terminated or truncated)
                    else (pre_action if agent == moving_agent else fixed_action)
                )
                env.step(action_to_take)
                if agent == order[-1]:
                    break
            if env.terminations.get(moving_agent) or env.truncations.get(moving_agent):
                break
    for _ in range(steps):
        for agent in env.agent_iter():
            observation, _reward, terminated, truncated, _info = env.last()
            if agent == moving_agent:
                frames.append(observation.copy())
            action_to_take = (
                None
                if (terminated or truncated)
                else (action if agent == moving_agent else fixed_action)
            )
            env.step(action_to_take)
            if agent == order[-1]:
                break
        if env.terminations.get(moving_agent) or env.truncations.get(moving_agent):
            break
        for _ in range(noop_every):
            for agent in env.agent_iter():
                observation, _reward, terminated, truncated, _info = env.last()
                if agent == moving_agent:
                    frames.append(observation.copy())
                action_to_take = None if (terminated or truncated) else fixed_action
                env.step(action_to_take)
                if agent == order[-1]:
                    break
            if env.terminations.get(moving_agent) or env.truncations.get(moving_agent):
                break
        if env.terminations.get(moving_agent) or env.truncations.get(moving_agent):
            break
    return np.asarray(frames, dtype=np.uint8)


def run_multiagent_sweep_trials(
    env,
    moving_agent: str,
    action: int,
    steps: int,
    seed: int,
    fixed_action: int,
    noop_every: int,
    pre_actions: Sequence[int] | None = None,
    trials: int = 1,
    expected_sign: int | None = None,
) -> np.ndarray:
    best_frames: np.ndarray | None = None
    best_score = -1.0
    for trial in range(trials):
        values = run_multiagent_sweep(
            env,
            moving_agent=moving_agent,
            action=action,
            steps=steps,
            seed=seed + (trial * 1000),
            fixed_action=fixed_action,
            noop_every=noop_every,
            pre_actions=pre_actions,
        )
        if not values.size:
            continue
        stats = rank_candidates(values, top_n=1)
        if not stats:
            continue
        score = abs(stats[0].corr)
        if expected_sign is not None and np.sign(stats[0].corr) != expected_sign:
            score *= 0.5
        if score > best_score:
            best_score = score
            best_frames = values
    if best_frames is None:
        return np.zeros((0, 128), dtype=np.uint8)
    return best_frames


def run_multiagent_pattern(
    env,
    moving_agent: str,
    actions: Sequence[int],
    seed: int,
    fixed_action: int,
    noop_every: int,
    pre_actions: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frames: list[np.ndarray] = []
    signals: list[int] = []
    reset_out = env.reset(seed=seed)
    observations = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    is_parallel = isinstance(observations, dict)
    if is_parallel:
        if moving_agent not in observations:
            raise ValueError(f"Unknown moving agent: {moving_agent}")
        if pre_actions:
            for pre_action in pre_actions:
                actions_dict = {
                    agent: (pre_action if agent == moving_agent else fixed_action)
                    for agent in env.agents
                }
                observations, _rewards, terminations, truncations, _infos = env.step(actions_dict)
                if terminations.get(moving_agent) or truncations.get(moving_agent):
                    break
        frames.append(observations[moving_agent].copy())
        signals.append(0)
        for action in actions:
            actions_dict = {
                agent: (action if agent == moving_agent else fixed_action) for agent in env.agents
            }
            observations, _rewards, terminations, truncations, _infos = env.step(actions_dict)
            if moving_agent in observations:
                frames.append(observations[moving_agent].copy())
                signals.append(action)
            if terminations.get(moving_agent) or truncations.get(moving_agent):
                break
            for _ in range(noop_every):
                actions_dict = dict.fromkeys(env.agents, fixed_action)
                observations, _rewards, terminations, truncations, _infos = env.step(actions_dict)
                if moving_agent in observations:
                    frames.append(observations[moving_agent].copy())
                    signals.append(0)
                if terminations.get(moving_agent) or truncations.get(moving_agent):
                    break
            if terminations.get(moving_agent) or truncations.get(moving_agent):
                break
        return np.asarray(frames, dtype=np.uint8), np.asarray(signals, dtype=np.int16)

    if not hasattr(env, "agent_iter"):
        raise ValueError("Unsupported PettingZoo environment interface.")

    if moving_agent not in env.possible_agents:
        raise ValueError(f"Unknown moving agent: {moving_agent}")

    order = env.possible_agents
    if pre_actions:
        for pre_action in pre_actions:
            for agent in env.agent_iter():
                observation, _reward, terminated, truncated, _info = env.last()
                action_to_take = (
                    None
                    if (terminated or truncated)
                    else (pre_action if agent == moving_agent else fixed_action)
                )
                env.step(action_to_take)
                if agent == order[-1]:
                    break
            if env.terminations.get(moving_agent) or env.truncations.get(moving_agent):
                break
    frames.append(observations[moving_agent].copy())
    signals.append(0)
    for action in actions:
        for agent in env.agent_iter():
            observation, _reward, terminated, truncated, _info = env.last()
            if agent == moving_agent:
                frames.append(observation.copy())
                signals.append(action)
            action_to_take = (
                None
                if (terminated or truncated)
                else (action if agent == moving_agent else fixed_action)
            )
            env.step(action_to_take)
            if agent == order[-1]:
                break
        if env.terminations.get(moving_agent) or env.truncations.get(moving_agent):
            break
        for _ in range(noop_every):
            for agent in env.agent_iter():
                observation, _reward, terminated, truncated, _info = env.last()
                if agent == moving_agent:
                    frames.append(observation.copy())
                    signals.append(0)
                action_to_take = None if (terminated or truncated) else fixed_action
                env.step(action_to_take)
                if agent == order[-1]:
                    break
            if env.terminations.get(moving_agent) or env.truncations.get(moving_agent):
                break
        if env.terminations.get(moving_agent) or env.truncations.get(moving_agent):
            break
    return np.asarray(frames, dtype=np.uint8), np.asarray(signals, dtype=np.int16)


def run_multiagent_reset_sweep(
    env,
    moving_agent: str,
    action: int,
    steps: int,
    seed: int,
    fixed_action: int,
    pre_actions: Sequence[int] | None = None,
) -> np.ndarray:
    frames: list[np.ndarray] = []
    for step_count in range(1, steps + 1):
        reset_out = env.reset(seed=seed)
        observations = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        is_parallel = isinstance(observations, dict)
        if is_parallel:
            if pre_actions:
                for pre_action in pre_actions:
                    actions_dict = {
                        agent: (pre_action if agent == moving_agent else fixed_action)
                        for agent in env.agents
                    }
                    observations, _rewards, terminations, truncations, _infos = env.step(
                        actions_dict
                    )
                    if terminations.get(moving_agent) or truncations.get(moving_agent):
                        break
            for _ in range(step_count):
                actions_dict = {
                    agent: (action if agent == moving_agent else fixed_action)
                    for agent in env.agents
                }
                observations, _rewards, terminations, truncations, _infos = env.step(actions_dict)
                if terminations.get(moving_agent) or truncations.get(moving_agent):
                    break
            if moving_agent in observations:
                frames.append(observations[moving_agent].copy())
            continue

        if not hasattr(env, "agent_iter"):
            raise ValueError("Unsupported PettingZoo environment interface.")

        order = env.possible_agents
        last_obs = None
        if pre_actions:
            for pre_action in pre_actions:
                for agent in env.agent_iter():
                    observation, _reward, terminated, truncated, _info = env.last()
                    if agent == moving_agent:
                        last_obs = observation.copy()
                    action_to_take = (
                        None
                        if (terminated or truncated)
                        else (pre_action if agent == moving_agent else fixed_action)
                    )
                    env.step(action_to_take)
                    if agent == order[-1]:
                        break
                if env.terminations.get(moving_agent) or env.truncations.get(moving_agent):
                    break
        for _ in range(step_count):
            for agent in env.agent_iter():
                observation, _reward, terminated, truncated, _info = env.last()
                if agent == moving_agent:
                    last_obs = observation.copy()
                action_to_take = (
                    None
                    if (terminated or truncated)
                    else (action if agent == moving_agent else fixed_action)
                )
                env.step(action_to_take)
                if agent == order[-1]:
                    break
            if env.terminations.get(moving_agent) or env.truncations.get(moving_agent):
                break
        if last_obs is not None:
            frames.append(last_obs)
    return np.asarray(frames, dtype=np.uint8)


def _detect_head(prev_frame: np.ndarray, frame: np.ndarray) -> tuple[int, int] | None:
    diff = np.abs(frame.astype(np.int16) - prev_frame.astype(np.int16))
    mask = diff.sum(axis=2) > 0
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    idx = np.argmax(diff[ys, xs].sum(axis=1))
    return int(xs[idx]), int(ys[idx])


def run_multiagent_visual_sweep(
    env,
    moving_agent: str,
    actions: Sequence[int],
    seed: int,
    fixed_action: int,
    pre_actions: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reset_out = env.reset(seed=seed)
    observations = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    if not isinstance(observations, dict):
        raise ValueError("Visual sweep requires parallel PettingZoo env.")

    if pre_actions:
        for pre_action in pre_actions:
            actions_dict = {
                agent: (pre_action if agent == moving_agent else fixed_action)
                for agent in env.agents
            }
            observations, _rewards, terminations, truncations, _infos = env.step(actions_dict)
            if terminations.get(moving_agent) or truncations.get(moving_agent):
                break

    prev_frame = env.render()
    if prev_frame is None:
        raise ValueError("Render frame unavailable; ensure render_mode='rgb_array'.")

    ram_frames: list[np.ndarray] = []
    xs: list[int] = []
    ys: list[int] = []
    for action in actions:
        actions_dict = {
            agent: (action if agent == moving_agent else fixed_action) for agent in env.agents
        }
        observations, _rewards, terminations, truncations, _infos = env.step(actions_dict)
        frame = env.render()
        if frame is None:
            break
        head = _detect_head(prev_frame, frame)
        prev_frame = frame
        if head is None:
            continue
        x, y = head
        if moving_agent in observations:
            ram_frames.append(observations[moving_agent].copy())
            xs.append(x)
            ys.append(y)
        if terminations.get(moving_agent) or truncations.get(moving_agent):
            break

    if not ram_frames:
        return np.zeros((0, 128), dtype=np.uint8), np.zeros(0), np.zeros(0)
    return np.asarray(ram_frames, dtype=np.uint8), np.asarray(xs), np.asarray(ys)


def _safe_corr(series: np.ndarray, steps: np.ndarray) -> float:
    if series.size < 2 or np.std(series) == 0 or np.std(steps) == 0:
        return 0.0
    corr = np.corrcoef(series, steps)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _trend_score(series: np.ndarray) -> float:
    if series.size < 2:
        return 0.0
    delta = np.diff(series.astype(np.int16))
    if delta.size == 0:
        return 0.0
    return float(np.mean(np.sign(delta)))


def rank_candidates(values: np.ndarray, top_n: int) -> list[CandidateStat]:
    steps = np.arange(values.shape[0], dtype=np.float32)
    stats: list[CandidateStat] = []
    for idx in range(values.shape[1]):
        series = values[:, idx].astype(np.float32)
        corr = _safe_corr(series, steps)
        trend = _trend_score(series)
        value_min = int(series.min())
        value_max = int(series.max())
        stats.append(
            CandidateStat(
                index=idx,
                corr=corr,
                trend=trend,
                value_min=value_min,
                value_max=value_max,
            )
        )
    stats.sort(
        key=lambda s: (abs(s.corr), abs(s.trend), s.value_max - s.value_min),
        reverse=True,
    )
    return stats[:top_n]


def rank_candidates_with_signal(
    values: np.ndarray,
    signal: np.ndarray,
    top_n: int,
) -> list[CandidateStat]:
    stats: list[CandidateStat] = []
    mask = signal != 0
    if mask.sum() < 2:
        return []
    filtered_signal = signal[mask].astype(np.float32)
    for idx in range(values.shape[1]):
        series = values[:, idx].astype(np.float32)
        filtered_series = series[mask]
        corr = _safe_corr(filtered_series, filtered_signal)
        trend = _trend_score(filtered_series)
        value_min = int(filtered_series.min())
        value_max = int(filtered_series.max())
        stats.append(
            CandidateStat(
                index=idx,
                corr=corr,
                trend=trend,
                value_min=value_min,
                value_max=value_max,
            )
        )
    stats.sort(key=lambda s: (abs(s.corr), abs(s.trend), s.value_max - s.value_min), reverse=True)
    return stats[:top_n]


def print_candidates(
    title: str,
    stats: Iterable[CandidateStat],
    corr_sign: int | None = None,
) -> None:
    print(f"\n{title}")
    print("index  corr    trend   min  max  range")
    for stat in stats:
        if corr_sign is not None and np.sign(stat.corr) != corr_sign:
            continue
        value_range = stat.value_max - stat.value_min
        print(
            f"{stat.index:>5}  {stat.corr:>5.2f}  {stat.trend:>6.2f}  "
            f"{stat.value_min:>3}  {stat.value_max:>3}  {value_range:>5}"
        )


def summarize_intersection(
    right_stats: Sequence[CandidateStat],
    left_stats: Sequence[CandidateStat],
    up_stats: Sequence[CandidateStat],
    down_stats: Sequence[CandidateStat],
) -> None:
    right_pos = {s.index for s in right_stats if s.corr > 0}
    left_neg = {s.index for s in left_stats if s.corr < 0}
    up_neg = {s.index for s in up_stats if s.corr < 0}
    down_pos = {s.index for s in down_stats if s.corr > 0}

    print("\nCross-check candidates (directional agreement)")
    print(f"x_candidates (RIGHT+, LEFT-): {sorted(right_pos & left_neg)}")
    print(f"y_candidates (UP-, DOWN+): {sorted(up_neg & down_pos)}")


def compute_distances(
    self_x: int,
    self_y: int,
    opp_x: int,
    opp_y: int,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> tuple[int, int, int, int, int, int]:
    left = self_x - x_min
    right = x_max - self_x
    up = self_y - y_min
    down = y_max - self_y
    dx = opp_x - self_x
    dy = opp_y - self_y
    return left, right, up, down, dx, dy


def extract_features_from_ram(
    ram: np.ndarray,
    indices: PositionIndices,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> tuple[int, int, int, int, int, int]:
    self_x = int(ram[indices.self_x])
    self_y = int(ram[indices.self_y])
    opp_x = int(ram[indices.opp_x])
    opp_y = int(ram[indices.opp_y])
    return compute_distances(self_x, self_y, opp_x, opp_y, x_min, x_max, y_min, y_max)


def _pick_axis_index(
    positive_stats: Sequence[CandidateStat],
    negative_stats: Sequence[CandidateStat],
) -> int:
    positives = {s.index: s for s in positive_stats if s.corr > 0}
    negatives = {s.index: s for s in negative_stats if s.corr < 0}
    shared = positives.keys() & negatives.keys()
    if shared:
        scored = []
        for idx in shared:
            score = abs(positives[idx].corr) + abs(negatives[idx].corr)
            scored.append((score, idx))
        scored.sort(reverse=True)
        return scored[0][1]
    ranked = positive_stats + negative_stats
    ranked.sort(key=lambda s: abs(s.corr), reverse=True)
    return ranked[0].index


def discover_position_indices(
    env,
    steps: int,
    seed: int,
    top_n: int,
    noop_every: int,
) -> tuple[PositionIndices, Bounds]:
    sweeps = [
        SweepConfig("RIGHT", ACTION_IDS["RIGHT"], steps),
        SweepConfig("LEFT", ACTION_IDS["LEFT"], steps),
        SweepConfig("UP", ACTION_IDS["UP"], steps),
        SweepConfig("DOWN", ACTION_IDS["DOWN"], steps),
    ]

    sweep_data: dict[str, np.ndarray] = {}
    for idx, sweep in enumerate(sweeps):
        pre_actions = None
        if sweep.name == "LEFT":
            pre_actions = [ACTION_IDS["UP"]] * PRE_TURN_STEPS + [
                ACTION_IDS["LEFT"]
            ] * PRE_TURN_STEPS
        values = run_sweep(
            env,
            sweep.action,
            sweep.steps,
            seed=seed + idx,
            noop_every=noop_every,
            pre_actions=pre_actions,
        )
        sweep_data[sweep.name] = values

    right_stats = rank_candidates(sweep_data["RIGHT"], top_n)
    left_stats = rank_candidates(sweep_data["LEFT"], top_n)
    up_stats = rank_candidates(sweep_data["UP"], top_n)
    down_stats = rank_candidates(sweep_data["DOWN"], top_n)

    self_x = _pick_axis_index(right_stats, left_stats)
    self_y = _pick_axis_index(down_stats, up_stats)

    x_values = np.concatenate([sweep_data["RIGHT"][:, self_x], sweep_data["LEFT"][:, self_x]])
    y_values = np.concatenate([sweep_data["UP"][:, self_y], sweep_data["DOWN"][:, self_y]])
    bounds = Bounds(
        x_min=int(x_values.min()),
        x_max=int(x_values.max()),
        y_min=int(y_values.min()),
        y_max=int(y_values.max()),
    )

    noop_values = run_sweep(
        env,
        ACTION_IDS["NOOP"],
        steps,
        seed=seed + 99,
        noop_every=noop_every,
    )
    variances = np.var(noop_values.astype(np.float32), axis=0)
    candidate_order = np.argsort(variances)[::-1].tolist()
    candidate_order = [idx for idx in candidate_order if idx not in {self_x, self_y}]
    opp_x = candidate_order[0] if len(candidate_order) > 0 else self_x
    opp_y = candidate_order[1] if len(candidate_order) > 1 else self_y

    indices = PositionIndices(self_x=self_x, self_y=self_y, opp_x=opp_x, opp_y=opp_y)
    return indices, bounds


def _direction_sweeps(steps: int) -> list[SweepConfig]:
    return [
        SweepConfig("RIGHT", ACTION_IDS["RIGHT"], steps),
        SweepConfig("LEFT", ACTION_IDS["LEFT"], steps),
        SweepConfig("UP", ACTION_IDS["UP"], steps),
        SweepConfig("DOWN", ACTION_IDS["DOWN"], steps),
    ]


def discover_multiagent_indices(
    steps: int,
    seed: int,
    top_n: int,
    difficulty: int,
    mode: int,
    fixed_action: int,
    noop_every: int,
    seed_trials: int = 5,
) -> tuple[PositionIndices, Bounds, dict[str, list[CandidateStat]]]:
    env = make_multiagent_env(difficulty, mode)
    try:
        agents = env.possible_agents if hasattr(env, "possible_agents") else env.agents
        if not agents:
            raise ValueError("Multi-agent env has no agents.")
        primary = agents[0]
        opponent = agents[1] if len(agents) > 1 else agents[0]

        sweep_data: dict[str, np.ndarray] = {}
        for idx, sweep in enumerate(_direction_sweeps(steps)):
            expected_sign = 1 if sweep.name in ("RIGHT", "DOWN") else -1
            left_pre_actions = [ACTION_IDS["UP"]] * PRE_TURN_STEPS + [
                ACTION_IDS["LEFT"]
            ] * PRE_TURN_STEPS
            values = run_multiagent_sweep_trials(
                env,
                moving_agent=primary,
                action=sweep.action,
                steps=sweep.steps,
                seed=seed + idx,
                fixed_action=fixed_action,
                noop_every=noop_every,
                pre_actions=left_pre_actions if sweep.name == "LEFT" else None,
                trials=seed_trials,
                expected_sign=expected_sign,
            )
            sweep_data[sweep.name] = values

        right_stats = rank_candidates(sweep_data["RIGHT"], top_n)
        left_stats = rank_candidates(sweep_data["LEFT"], top_n)
        up_stats = rank_candidates(sweep_data["UP"], top_n)
        down_stats = rank_candidates(sweep_data["DOWN"], top_n)

        self_x_candidates = right_stats
        self_x = _pick_axis_index(right_stats, left_stats)
        self_y = _pick_axis_index(down_stats, up_stats)

        right_pos = {s.index for s in right_stats if s.corr > 0}
        left_neg = {s.index for s in left_stats if s.corr < 0}
        if not (right_pos & left_neg):
            reset_right = run_multiagent_reset_sweep(
                env,
                moving_agent=primary,
                action=ACTION_IDS["RIGHT"],
                steps=steps,
                seed=seed + 400,
                fixed_action=fixed_action,
                pre_actions=(
                    [ACTION_IDS["UP"]] * PRE_TURN_STEPS + [ACTION_IDS["RIGHT"]] * PRE_TURN_STEPS
                ),
            )
            reset_left = run_multiagent_reset_sweep(
                env,
                moving_agent=primary,
                action=ACTION_IDS["LEFT"],
                steps=steps,
                seed=seed + 401,
                fixed_action=fixed_action,
                pre_actions=(
                    [ACTION_IDS["UP"]] * PRE_TURN_STEPS + [ACTION_IDS["LEFT"]] * PRE_TURN_STEPS
                ),
            )
            reset_right_stats = rank_candidates(reset_right, top_n)
            reset_left_stats = rank_candidates(reset_left, top_n)
            if reset_right_stats and reset_left_stats:
                self_x_candidates = reset_right_stats
                self_x = _pick_axis_index(reset_right_stats, reset_left_stats)

            pattern_actions = [ACTION_IDS["RIGHT"], ACTION_IDS["LEFT"]] * steps
            pattern_values, pattern_signal = run_multiagent_pattern(
                env,
                moving_agent=primary,
                actions=pattern_actions,
                seed=seed + 200,
                fixed_action=fixed_action,
                noop_every=noop_every,
                pre_actions=[ACTION_IDS["UP"]] * PRE_TURN_STEPS,
            )
            signal = np.where(
                pattern_signal == ACTION_IDS["RIGHT"],
                1,
                np.where(pattern_signal == ACTION_IDS["LEFT"], -1, 0),
            )
            self_x_candidates = rank_candidates_with_signal(pattern_values, signal, top_n)
            if self_x_candidates:
                self_x = self_x_candidates[0].index

        if not (right_pos & left_neg):
            env_visual = make_multiagent_env(difficulty, mode, render_mode="rgb_array")
            try:
                actions = (
                    [ACTION_IDS["RIGHT"]] * steps
                    + [ACTION_IDS["LEFT"]] * steps
                    + [ACTION_IDS["DOWN"]] * steps
                    + [ACTION_IDS["UP"]] * steps
                )
                ram_values, xs, ys = run_multiagent_visual_sweep(
                    env_visual,
                    moving_agent=primary,
                    actions=actions,
                    seed=seed + 700,
                    fixed_action=fixed_action,
                    pre_actions=[ACTION_IDS["UP"]] * PRE_TURN_STEPS,
                )
                if ram_values.size and xs.size:
                    x_stats = rank_candidates_with_signal(ram_values, xs + 1, top_n)
                    if x_stats:
                        self_x_candidates = x_stats
                        self_x = x_stats[0].index
                        visual_self_x_stats = x_stats
            finally:
                env_visual.close()

        x_values = np.concatenate([sweep_data["RIGHT"][:, self_x], sweep_data["LEFT"][:, self_x]])
        y_values = np.concatenate([sweep_data["UP"][:, self_y], sweep_data["DOWN"][:, self_y]])
        bounds = Bounds(
            x_min=int(x_values.min()),
            x_max=int(x_values.max()),
            y_min=int(y_values.min()),
            y_max=int(y_values.max()),
        )

        opp_sweep_data: dict[str, np.ndarray] = {}
        for idx, sweep in enumerate(_direction_sweeps(steps)):
            expected_sign = 1 if sweep.name in ("RIGHT", "DOWN") else -1
            left_pre_actions = [ACTION_IDS["UP"]] * PRE_TURN_STEPS + [
                ACTION_IDS["LEFT"]
            ] * PRE_TURN_STEPS
            values = run_multiagent_sweep_trials(
                env,
                moving_agent=opponent,
                action=sweep.action,
                steps=sweep.steps,
                seed=seed + 100 + idx,
                fixed_action=fixed_action,
                noop_every=noop_every,
                pre_actions=left_pre_actions if sweep.name == "LEFT" else None,
                trials=seed_trials,
                expected_sign=expected_sign,
            )
            opp_sweep_data[sweep.name] = values

        opp_right_stats = rank_candidates(opp_sweep_data["RIGHT"], top_n)
        opp_left_stats = rank_candidates(opp_sweep_data["LEFT"], top_n)
        opp_up_stats = rank_candidates(opp_sweep_data["UP"], top_n)
        opp_down_stats = rank_candidates(opp_sweep_data["DOWN"], top_n)

        opp_x_candidates = opp_right_stats
        opp_x = _pick_axis_index(opp_right_stats, opp_left_stats)
        opp_y = _pick_axis_index(opp_down_stats, opp_up_stats)

        opp_right_pos = {s.index for s in opp_right_stats if s.corr > 0}
        opp_left_neg = {s.index for s in opp_left_stats if s.corr < 0}
        if not (opp_right_pos & opp_left_neg):
            reset_right = run_multiagent_reset_sweep(
                env,
                moving_agent=opponent,
                action=ACTION_IDS["RIGHT"],
                steps=steps,
                seed=seed + 500,
                fixed_action=fixed_action,
                pre_actions=(
                    [ACTION_IDS["UP"]] * PRE_TURN_STEPS + [ACTION_IDS["RIGHT"]] * PRE_TURN_STEPS
                ),
            )
            reset_left = run_multiagent_reset_sweep(
                env,
                moving_agent=opponent,
                action=ACTION_IDS["LEFT"],
                steps=steps,
                seed=seed + 501,
                fixed_action=fixed_action,
                pre_actions=(
                    [ACTION_IDS["UP"]] * PRE_TURN_STEPS + [ACTION_IDS["LEFT"]] * PRE_TURN_STEPS
                ),
            )
            reset_right_stats = rank_candidates(reset_right, top_n)
            reset_left_stats = rank_candidates(reset_left, top_n)
            if reset_right_stats and reset_left_stats:
                opp_x_candidates = reset_right_stats
                opp_x = _pick_axis_index(reset_right_stats, reset_left_stats)

            pattern_actions = [ACTION_IDS["RIGHT"], ACTION_IDS["LEFT"]] * steps
            pattern_values, pattern_signal = run_multiagent_pattern(
                env,
                moving_agent=opponent,
                actions=pattern_actions,
                seed=seed + 300,
                fixed_action=fixed_action,
                noop_every=noop_every,
                pre_actions=[ACTION_IDS["UP"]] * PRE_TURN_STEPS,
            )
            signal = np.where(
                pattern_signal == ACTION_IDS["RIGHT"],
                1,
                np.where(pattern_signal == ACTION_IDS["LEFT"], -1, 0),
            )
            opp_x_candidates = rank_candidates_with_signal(pattern_values, signal, top_n)
            if opp_x_candidates:
                opp_x = opp_x_candidates[0].index

        if not (opp_right_pos & opp_left_neg):
            env_visual = make_multiagent_env(difficulty, mode, render_mode="rgb_array")
            try:
                actions = (
                    [ACTION_IDS["RIGHT"]] * steps
                    + [ACTION_IDS["LEFT"]] * steps
                    + [ACTION_IDS["DOWN"]] * steps
                    + [ACTION_IDS["UP"]] * steps
                )
                ram_values, xs, ys = run_multiagent_visual_sweep(
                    env_visual,
                    moving_agent=opponent,
                    actions=actions,
                    seed=seed + 800,
                    fixed_action=fixed_action,
                    pre_actions=[ACTION_IDS["UP"]] * PRE_TURN_STEPS,
                )
                if ram_values.size and xs.size:
                    x_stats = rank_candidates_with_signal(ram_values, xs + 1, top_n)
                    if x_stats:
                        opp_x_candidates = x_stats
                        opp_x = x_stats[0].index
                        visual_opp_x_stats = x_stats
            finally:
                env_visual.close()

        indices = PositionIndices(
            self_x=self_x,
            self_y=self_y,
            opp_x=opp_x,
            opp_y=opp_y,
        )
        stats = {
            "self_right": right_stats,
            "self_left": left_stats,
            "self_up": up_stats,
            "self_down": down_stats,
            "self_x_osc": self_x_candidates,
            "opp_right": opp_right_stats,
            "opp_left": opp_left_stats,
            "opp_up": opp_up_stats,
            "opp_down": opp_down_stats,
            "opp_x_osc": opp_x_candidates,
        }
        if "visual_self_x_stats" in locals():
            stats["self_x_visual"] = visual_self_x_stats
        if "visual_opp_x_stats" in locals():
            stats["opp_x_visual"] = visual_opp_x_stats
        return indices, bounds, stats
    finally:
        env.close()


def discover_and_build_extractor(
    steps: int = 60,
    seed: int = 0,
    top_n: int = 8,
    difficulty: int = 0,
    mode: int = 0,
    *,
    use_multi_agent: bool = False,
    noop_every: int = 1,
):
    """
    Runs discovery and returns a feature extractor closure.

    This uses action sweeps to find self position bytes and a NOOP sweep to
    guess opponent bytes by variance. The opponent detection is heuristic and
    should be validated after inspection.
    """
    if use_multi_agent:
        indices, bounds, _stats = discover_multiagent_indices(
            steps=steps,
            seed=seed,
            top_n=top_n,
            difficulty=difficulty,
            mode=mode,
            fixed_action=ACTION_IDS["NOOP"],
            noop_every=noop_every,
        )
    else:
        env = make_env(difficulty, mode)
        try:
            indices, bounds = discover_position_indices(
                env,
                steps=steps,
                seed=seed,
                top_n=top_n,
                noop_every=noop_every,
            )
        finally:
            env.close()

    def extractor(ram: np.ndarray) -> tuple[int, int, int, int, int, int]:
        return extract_features_from_ram(
            ram,
            indices=indices,
            x_min=bounds.x_min,
            x_max=bounds.x_max,
            y_min=bounds.y_min,
            y_max=bounds.y_max,
        )

    return indices, bounds, extractor


def create_extractor(
    steps: int = 60,
    seed: int = 0,
    top_n: int = 8,
    difficulty: int = 0,
    mode: int = 0,
    *,
    use_multi_agent: bool = False,
    noop_every: int = 1,
):
    """
    Convenience wrapper that returns only the feature extractor.
    """
    _indices, _bounds, extractor = discover_and_build_extractor(
        steps=steps,
        seed=seed,
        top_n=top_n,
        difficulty=difficulty,
        mode=mode,
        use_multi_agent=use_multi_agent,
        noop_every=noop_every,
    )
    return extractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe ALE RAM for Surround position bytes.")
    parser.add_argument("--steps", type=int, default=60, help="Steps per sweep direction.")
    parser.add_argument("--top", type=int, default=8, help="Top N RAM indices to report per sweep.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic resets.")
    parser.add_argument("--difficulty", type=int, default=0, help="ALE difficulty.")
    parser.add_argument("--mode", type=int, default=0, help="ALE mode.")
    parser.add_argument(
        "--noop-every",
        type=int,
        default=1,
        help="Number of NOOP steps to insert after each action.",
    )
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Run multi-agent validation sweeps (PettingZoo required).",
    )
    parser.add_argument(
        "--seed-trials",
        type=int,
        default=5,
        help="Number of seed trials to concatenate per sweep.",
    )
    args = parser.parse_args()

    _extractor = create_extractor(
        steps=args.steps,
        seed=args.seed,
        top_n=args.top,
        difficulty=args.difficulty,
        mode=args.mode,
        use_multi_agent=args.multi_agent,
        noop_every=args.noop_every,
    )

    if args.multi_agent:
        indices, bounds, stats = discover_multiagent_indices(
            steps=args.steps,
            seed=args.seed,
            top_n=args.top,
            difficulty=args.difficulty,
            mode=args.mode,
            fixed_action=ACTION_IDS["NOOP"],
            noop_every=args.noop_every,
            seed_trials=args.seed_trials,
        )
        print("\nMulti-agent validation (agent0 moving, agent1 NOOP)")
        print_candidates("agent0 RIGHT", stats["self_right"], corr_sign=1)
        print_candidates("agent0 LEFT", stats["self_left"], corr_sign=-1)
        print_candidates("agent0 UP", stats["self_up"], corr_sign=-1)
        print_candidates("agent0 DOWN", stats["self_down"], corr_sign=1)
        print_candidates("agent0 X oscillation", stats["self_x_osc"])
        if "self_x_visual" in stats:
            print_candidates("agent0 X visual", stats["self_x_visual"])
        summarize_intersection(
            stats["self_right"],
            stats["self_left"],
            stats["self_up"],
            stats["self_down"],
        )
        print("\nMulti-agent validation (agent1 moving, agent0 NOOP)")
        print_candidates("agent1 RIGHT", stats["opp_right"], corr_sign=1)
        print_candidates("agent1 LEFT", stats["opp_left"], corr_sign=-1)
        print_candidates("agent1 UP", stats["opp_up"], corr_sign=-1)
        print_candidates("agent1 DOWN", stats["opp_down"], corr_sign=1)
        print_candidates("agent1 X oscillation", stats["opp_x_osc"])
        if "opp_x_visual" in stats:
            print_candidates("agent1 X visual", stats["opp_x_visual"])
        summarize_intersection(
            stats["opp_right"],
            stats["opp_left"],
            stats["opp_up"],
            stats["opp_down"],
        )
        print(f"\nSelected indices (multi-agent): {indices}")
        print(f"Bounds (multi-agent): {bounds}")
        return

    env = make_env(args.difficulty, args.mode)
    try:
        sweeps = [
            SweepConfig("RIGHT", ACTION_IDS["RIGHT"], args.steps),
            SweepConfig("LEFT", ACTION_IDS["LEFT"], args.steps),
            SweepConfig("UP", ACTION_IDS["UP"], args.steps),
            SweepConfig("DOWN", ACTION_IDS["DOWN"], args.steps),
        ]

        sweep_data: dict[str, np.ndarray] = {}
        for idx, sweep in enumerate(sweeps):
            values = run_sweep(
                env,
                sweep.action,
                sweep.steps,
                seed=args.seed + idx,
                noop_every=args.noop_every,
            )
            sweep_data[sweep.name] = values

        right_stats = rank_candidates(sweep_data["RIGHT"], args.top)
        left_stats = rank_candidates(sweep_data["LEFT"], args.top)
        up_stats = rank_candidates(sweep_data["UP"], args.top)
        down_stats = rank_candidates(sweep_data["DOWN"], args.top)

        print_candidates("RIGHT sweep candidates (expect x to increase)", right_stats, corr_sign=1)
        print_candidates("LEFT sweep candidates (expect x to decrease)", left_stats, corr_sign=-1)
        print_candidates("UP sweep candidates (expect y to decrease)", up_stats, corr_sign=-1)
        print_candidates("DOWN sweep candidates (expect y to increase)", down_stats, corr_sign=1)
        summarize_intersection(right_stats, left_stats, up_stats, down_stats)
    finally:
        env.close()


if __name__ == "__main__":
    main()
