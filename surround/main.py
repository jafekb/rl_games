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

env = surround_v2.env(
    obs_type="ram",
    full_action_space=True,
    max_cycles=MAX_CYCLES,
    auto_rom_install_path=ROM_PATH,
    render_mode="rgb_array",
)

VIDEO_DIR.mkdir(parents=True, exist_ok=True)

print(f"Human agent: {HUMAN_AGENT} | AI agent: {AI_AGENT}")

action_meanings = None
if hasattr(env.unwrapped, "get_action_meanings"):
    action_meanings = env.unwrapped.get_action_meanings()


def get_human_action(action_space, action_names):
    if action_names and "RIGHT" in action_names:
        return action_names.index("RIGHT")
    return 0


def get_ai_action(action_space, action_names):
    if action_names:
        directional = [
            action_names.index(name)
            for name in ("UP", "RIGHT", "LEFT", "DOWN")
            if name in action_names
        ]
        if directional:
            return directional[action_space.sample() % len(directional)]
    return action_space.sample()


env.reset()
video_writer = imageio.get_writer(str(VIDEO_PATH), fps=VIDEO_FPS, macro_block_size=1)
if action_meanings:
    print("Action meanings:", action_meanings)
try:
    for cycle_step in trange(MAX_CYCLES):
        agents_this_cycle = list(env.agents)
        if not agents_this_cycle:
            break
        for _ in agents_this_cycle:
            agent = env.agent_selection
            observation, reward, termination, truncation, info = env.last()
            # this is where you would insert your policy
            if termination or truncation:
                action = None
            elif agent == HUMAN_AGENT:
                action = get_human_action(env.action_space(agent), action_meanings)
            elif agent == AI_AGENT:
                action = get_ai_action(env.action_space(agent), action_meanings)
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
