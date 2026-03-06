"""Surround game and training configuration constants."""

from pathlib import Path

from surround.utils.checkpoint import CheckpointPaths

# Game / simulation
GRID_ROWS = 18
GRID_COLS = 38
EMPTY_CELL = 0
WALL_CELL = 1
EGO_CELL = 2
FRAME_SKIP = 4
DEBUG_STATE = False

# Env / run
DIFFICULTY = 1
MODE = 0
SEED = 0
MAX_CYCLES = 1000
CHECKPOINT_INTERVAL = 1000
NUM_EPISODES = 50_000

# Q-learning
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 1000
STEP_REWARD = 0.01
STATE_MODE = "state_tuple"
WINDOW_SIZE = 7

# Paths (Q-learning)
Q_TABLE_PATH = Path("surround/q_learning/q_table.json")
QL_LOG_DIR = Path("runs/surround/ql_visits")

# DQN
DQN_STATE_TYPE = "class_map"
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY_FRACTION = 0.01
TAU = 0.005
LR = 3e-4
MEMORY_CAPACITY = 10_000
GAME_ROW_SLICE = slice(35, 198)
GAME_COL_SLICE = slice(4, 156)
DQN_PREPROCESS_HEIGHT = 80
DQN_PREPROCESS_WIDTH = 80
N_ACTIONS = 4
DQN_LOG_DIR = Path("runs/surround/dqn/exp16")
DQN_EPISODE_VIDEO_FPS = 10
DQN_VISUALIZE_EPISODES = False
DQN_CKPT = CheckpointPaths(DQN_LOG_DIR)

# PPO (state-tuple input)
PPO_STATE_TUPLE_DIM = 7
PPO_LR = 3e-4
PPO_EPS_CLIP = 0.2
PPO_EPOCHS = 10
PPO_ENTROPY_COEF = 0.02
PPO_GRAD_CLIP = 0
PPO_UPDATE_TIMESTEP = 2000
PPO_LOG_DIR = Path("runs/surround/ppo/rollout2000_2k_ep")
PPO_CKPT = CheckpointPaths(PPO_LOG_DIR)

# D3QN (Dueling Double DQN, uniform replay, n-step returns)
D3QN_LR = 1e-4
D3QN_N_STEP = 10
D3QN_LEARNING_STARTS = 5_000
D3QN_UPDATE_EVERY = 4
D3QN_MEMORY_CAPACITY = 100_000
D3QN_BATCH_SIZE = 256
D3QN_EPS_DECAY_FRACTION = 0.05
D3QN_LOG_DIR = Path("runs/surround/d3qn/exp10")
D3QN_CKPT = CheckpointPaths(D3QN_LOG_DIR)
D3QN_RESUME_FROM: Path | None = Path("runs/surround/d3qn/exp9")
D3QN_FRESH_EPSILON: bool = True
D3QN_RANDOM_DIFFICULTY: bool = True
