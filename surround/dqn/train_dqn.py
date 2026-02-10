"""
Train a DQN agent for the Surround game.


See https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import math
import random
from collections import deque, namedtuple
from itertools import count

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

from surround.actions import ACTION_WORD_TO_ID
from surround.utils.video_extract_locations import get_location

DIFFICULTY = 0
MODE = 0
FRAME_SKIP = 4
GRID_ROWS = 18
GRID_COLS = 38
STATE_MODE = "state_tuple"

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPISODES = 600


def make_env():
    gym.register_envs(ale_py)
    return gym.make(
        "ALE/Surround-v5",
        obs_type="rgb",
        full_action_space=False,
        difficulty=DIFFICULTY,
        mode=MODE,
        frameskip=FRAME_SKIP,
    )


def get_state_tuple(locations, last_action: int) -> tuple[int, ...]:
    """
    Builds a state tuple from the locations of the ego, opponent, and walls.

    Returns:
        A tuple (d_up, d_right, d_left, d_down, rel_x, rel_y, last_action).
        There are 2*2*2*2*3*3*4 = 576 possible states.
    """
    if locations["ego"] is None or locations["opp"] is None:
        return (1, 1, 1, 1, 1, 1, last_action)
    ego_row, ego_col = locations["ego"]
    opp_row, opp_col = locations["opp"]
    wall_set = locations["walls"]
    collisions = (
        wall_set | {(opp_row, opp_col)} if opp_row is not None and opp_col is not None else wall_set
    )

    d_up = 1 if (ego_row - 1, ego_col) in collisions or ego_row <= 0 else 0
    d_right = 1 if (ego_row, ego_col + 1) in collisions or ego_col >= GRID_COLS - 1 else 0
    d_left = 1 if (ego_row, ego_col - 1) in collisions or ego_col <= 0 else 0
    d_down = 1 if (ego_row + 1, ego_col) in collisions or ego_row >= GRID_ROWS - 1 else 0

    if opp_col < ego_col:
        rel_x = 0
    elif opp_col > ego_col:
        rel_x = 2
    else:
        rel_x = 1

    if opp_row < ego_row:
        rel_y = 0
    elif opp_row > ego_row:
        rel_y = 2
    else:
        rel_y = 1

    return (d_up, d_right, d_left, d_down, rel_x, rel_y, last_action)


def _get_state(observation: np.ndarray, last_action: int) -> tuple[int, ...]:
    """Build state tuple from raw observation (same logic as Q-learning)."""
    if STATE_MODE == "ram":
        raise ValueError("RAM state mode is not supported.")
    frame = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    locations = get_location(frame)
    return get_state_tuple(locations, last_action)


def state_to_tensor(state_tuple: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Convert state tuple to (1, n_observations) float tensor."""
    return torch.tensor([state_tuple], dtype=torch.float32, device=device)


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(torch.nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()

        self.layer1 = torch.nn.Linear(n_observations, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        # TODO(bjafek) try out gelu?
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)


# ignore NOOP
env = make_env()
n_actions = env.action_space.n - 1
# State is 7-tuple from _get_state (same as Q-learning)
n_observations = 7

policy_net = DQN(n_observations, n_actions).to(DEVICE)
target_net = DQN(n_observations, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10_000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # TODO(bjafek) probably have to increment by 1 to account for NOOP removal
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=DEVICE, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose to convert batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(
        tuple(s is not None for s in batch.next_state),
        device=DEVICE,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # And optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def train():
    episode_durations = []
    for episode_index in trange(NUM_EPISODES):
        observation, info = env.reset()
        last_action = ACTION_WORD_TO_ID["LEFT"]
        state_tuple = _get_state(observation, last_action)
        state = state_to_tensor(state_tuple, DEVICE)
        for t in count():
            action = select_action(state)
            action_id = action.item() + 1  # env expects 1..4 (no NOOP)
            observation, reward, terminated, truncated, _info = env.step(action_id)
            reward = torch.tensor([reward], device=DEVICE)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state_tuple = _get_state(observation, action_id)
                next_state = state_to_tensor(next_state_tuple, DEVICE)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization on the policy network
            optimize_model()

            # Soft update of the target networks' weights
            # theta' <- tau*theta + (1-tau)*theta'
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                # TODO(bjafek) log with tensorboard
                episode_durations.append(t + 1)
                break
    print("Training complete!")


train()
