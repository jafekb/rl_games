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
LOG_DIR = Path("runs/surround/ql_visits")

# DQN
DQN_FRAME_SKIP = 4
# Input type:
#     - "state_tuple": 7-tuple -> MLP)
#     - "grayscale": image -> CNN)
#     - "class_map": 4-class map -> CNN, exp11)
DQN_STATE_TYPE = "class_map"
BATCH_SIZE = 128
GAMMA_DQN = 0.99
EPS_START = 0.9
EPS_END = 0.01
# 0.01 so epsilon ~0.01 by ~ep 1000 (matches exp11 TB curve); 0.06 = step-based 100k@100steps/ep
EPS_DECAY_FRACTION = 0.005
TAU = 0.005
LR = 3e-4
MEMORY_CAPACITY = 10_000
NUM_EPISODES = 50_000
GAME_ROW_SLICE = slice(35, 198)
GAME_COL_SLICE = slice(4, 156)
DQN_GAME_HEIGHT = GAME_ROW_SLICE.stop - GAME_ROW_SLICE.start
DQN_GAME_WIDTH = GAME_COL_SLICE.stop - GAME_COL_SLICE.start
N_ACTIONS = 4
DQN_LOG_DIR = Path("runs/surround/dqn/exp11_4")
DQN_EPISODE_VIDEO_FPS = 10
VISUALIZE_EPISODES = False
DQN_CHECKPOINT_DIR = DQN_LOG_DIR / "checkpoints"
DQN_CHECKPOINT_INTERVAL = 50
DQN_POLICY_NET_LATEST = DQN_CHECKPOINT_DIR / "policy_net_latest.pt"
DQN_POLICY_NET_BEST = DQN_CHECKPOINT_DIR / "policy_net_best.pt"
DQN_CHECKPOINT_METADATA = DQN_CHECKPOINT_DIR / "metadata.json"
