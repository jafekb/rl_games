"""Surround game and training configuration constants."""

from pathlib import Path

# Game / simulation
GRID_ROWS = 18
GRID_COLS = 38
EMPTY_CELL = 0
WALL_CELL = 1
EGO_CELL = 2
FRAME_SKIP = 8
DEBUG_STATE = False

# Env / run
DIFFICULTY = 0
MODE = 0
SEED = 0
MAX_CYCLES = 10_000

# Q-learning
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 1000
EPISODES = 1_000_000
STEP_REWARD = 0.01
STATE_MODE = "state_tuple"
WINDOW_SIZE = 7

# Paths (Q-learning)
Q_TABLE_PATH = Path("surround/q_learning/q_table.json")
LOG_DIR = Path("runs/surround_ql_visits")

# DQN
DQN_FRAME_SKIP = 4
BATCH_SIZE = 128
GAMMA_DQN = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4
MEMORY_CAPACITY = 10_000
NUM_EPISODES = 600
N_OBSERVATIONS = 7  # state tuple size from get_state_tuple
N_ACTIONS = 4  # Surround without NOOP
DQN_LOG_DIR = Path("runs/surround_dqn_save")
DQN_CHECKPOINT_DIR = DQN_LOG_DIR / "checkpoints"
DQN_CHECKPOINT_INTERVAL = 50
DQN_POLICY_NET_LATEST = DQN_CHECKPOINT_DIR / "policy_net_latest.pt"
DQN_CHECKPOINT_METADATA = DQN_CHECKPOINT_DIR / "metadata.json"
