from pathlib import Path

import imageio.v2 as imageio
from pettingzoo.atari import surround_v2
from tqdm import trange

ROM_PATH = str(Path("~/.local/share/AutoROM/roms").expanduser())
MAX_CYCLES = 10000
VIDEO_DIR = Path("video")
VIDEO_PATH = VIDEO_DIR / "surround.mp4"
VIDEO_FPS = 120
FRAME_STRIDE = 4
HUMAN_AGENT = "second_0"
AI_AGENT = "first_0"
ACTION_WORDS_5 = ("NOOP", "UP", "RIGHT", "LEFT", "DOWN")
ACTION_WORD_TO_ID = {word: action_id for action_id, word in enumerate(ACTION_WORDS_5)}

env = surround_v2.env(
    obs_type="ram",
    full_action_space=False,
    max_cycles=MAX_CYCLES,
    auto_rom_install_path=ROM_PATH,
    render_mode="rgb_array",
)

VIDEO_DIR.mkdir(parents=True, exist_ok=True)

print(f"Human agent: {HUMAN_AGENT} | AI agent: {AI_AGENT}")


def get_human_action(action_space, action_names):
    return ACTION_WORD_TO_ID["RIGHT"]


def get_ai_action(action_space, action_names):
    directional = [ACTION_WORD_TO_ID[name] for name in ("UP", "RIGHT", "LEFT", "DOWN")]
    return directional[action_space.sample() % len(directional)]


env.reset()
video_writer = imageio.get_writer(str(VIDEO_PATH), fps=VIDEO_FPS, macro_block_size=1)
try:
    for cycle_step in trange(MAX_CYCLES):
        agents_this_cycle = list(env.agents)
        if not agents_this_cycle:
            break
        for _ in agents_this_cycle:
            agent = env.agent_selection
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            elif agent == HUMAN_AGENT:
                action = get_human_action(env.action_space(agent), None)
            elif agent == AI_AGENT:
                action = get_ai_action(env.action_space(agent), None)
            else:
                action = env.action_space(agent).sample()
            env.step(action)
        if cycle_step % FRAME_STRIDE == 0:
            frame = env.render()
            if frame is not None:
                video_writer.append_data(frame)
finally:
    video_writer.close()
    env.close()
