from pathlib import Path

import imageio.v2 as imageio
from pettingzoo.atari import surround_v2

ROM_PATH = str(Path("~/.local/share/AutoROM/roms").expanduser())
MAX_CYCLES = 100000
VIDEO_DIR = Path("video")
VIDEO_PATH = VIDEO_DIR / "surround.mp4"
VIDEO_FPS = 120
FRAME_STRIDE = 4

env = surround_v2.env(
    obs_type="ram",
    full_action_space=True,
    max_cycles=MAX_CYCLES,
    auto_rom_install_path=ROM_PATH,
    render_mode="rgb_array",
)

VIDEO_DIR.mkdir(parents=True, exist_ok=True)

env.reset()
video_writer = imageio.get_writer(str(VIDEO_PATH), fps=VIDEO_FPS)
try:
    for cycle_step in range(MAX_CYCLES):
        agents_this_cycle = list(env.agents)
        if not agents_this_cycle:
            break
        for _ in agents_this_cycle:
            agent = env.agent_selection
            observation, reward, termination, truncation, info = env.last()
            # this is where you would insert your policy
            action = None if termination or truncation else env.action_space(agent).sample()
            env.step(action)
        print(f"{cycle_step=}")
        if cycle_step % FRAME_STRIDE == 0:
            frame = env.render()
            if frame is not None:
                video_writer.append_data(frame)
finally:
    video_writer.close()
    env.close()
