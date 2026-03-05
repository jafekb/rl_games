"""Surround game and training configuration constants."""

from pathlib import Path

# Game / simulation
GRID_ROWS = 18
GRID_COLS = 38
EMPTY_CELL = 0
WALL_CELL = 1
EGO_CELL = 2
FRAME_SKIP = 4
DEBUG_STATE = False

# Env / run
DIFFICULTY = 0
MODE = 0
SEED = 0
MAX_CYCLES = 1_000
CHECKPOINT_INTERVAL = 1000

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
# Input type:
#     - "state_tuple": 7-tuple -> MLP
#     - "grayscale": image -> CNN
#     - "class_map": 4-class map -> CNN, exp11
DQN_STATE_TYPE = "class_map"
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY_FRACTION = 0.01
TAU = 0.005
LR = 3e-4
MEMORY_CAPACITY = 10_000
NUM_EPISODES = 50_000
GAME_ROW_SLICE = slice(35, 198)
GAME_COL_SLICE = slice(4, 156)
DQN_PREPROCESS_HEIGHT = 80
DQN_PREPROCESS_WIDTH = 80
N_ACTIONS = 4
DQN_LOG_DIR = Path("runs/surround/dqn/exp16")
DQN_EPISODE_VIDEO_FPS = 10
VISUALIZE_EPISODES = False
DQN_CHECKPOINT_DIR = DQN_LOG_DIR / "checkpoints"
DQN_POLICY_NET_LATEST = DQN_CHECKPOINT_DIR / "policy_net_latest.pt"
DQN_POLICY_NET_BEST = DQN_CHECKPOINT_DIR / "policy_net_best.pt"
DQN_CHECKPOINT_METADATA = DQN_CHECKPOINT_DIR / "metadata.json"

# PPO (state-tuple input)
STATE_TUPLE_DIM = 7
PPO_LR = 3e-4
PPO_EPS_CLIP = 0.2
PPO_EPOCHS = 10
PPO_ENTROPY_COEF = 0.02
PPO_GRAD_CLIP = 0  # 0 = disabled; use 0.5 for extra stability if needed
PPO_UPDATE_TIMESTEP = 2000
PPO_NUM_EPISODES = 2000
PPO_LOG_DIR = Path("runs/surround/ppo/rollout2000_2k_ep")
PPO_CHECKPOINT_DIR = PPO_LOG_DIR / "checkpoints"
PPO_POLICY_LATEST = PPO_CHECKPOINT_DIR / "policy_latest.pt"
PPO_POLICY_BEST = PPO_CHECKPOINT_DIR / "policy_best.pt"
PPO_CHECKPOINT_METADATA = PPO_CHECKPOINT_DIR / "metadata.json"

# D3QN (Dueling Double DQN, uniform replay, n-step returns)
# Hyperparameters reflect best-performing config (exp5).
D3QN_LR = 1e-4
D3QN_N_STEP = 10
D3QN_LEARNING_STARTS = 1_000
D3QN_UPDATE_EVERY = 4
D3QN_LOG_DIR = Path("runs/surround/d3qn/d3qn")
D3QN_CHECKPOINT_DIR = D3QN_LOG_DIR / "checkpoints"
D3QN_POLICY_NET_LATEST = D3QN_CHECKPOINT_DIR / "policy_net_latest.pt"
D3QN_POLICY_NET_BEST = D3QN_CHECKPOINT_DIR / "policy_net_best.pt"
D3QN_CHECKPOINT_METADATA = D3QN_CHECKPOINT_DIR / "metadata.json"
