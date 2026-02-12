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
CLIP_MAX = 7
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 1000
EPISODES = 1_000_000
STEP_REWARD = 0.01
STATE_MODE = "state_tuple"
WINDOW_SIZE = 7

# Paths
Q_TABLE_PATH = Path("surround/q_learning/q_table.json")
LOG_DIR = Path("runs/surround_ql_visits")
