from pathlib import Path

from gymnasium.wrappers import RecordVideo
from pettingzoo.atari import surround_v2

VIDEO_FOLDER = str(Path("surround_video").expanduser())
ROM_PATH = str(Path("~/.local/share/AutoROM/roms").expanduser())

env = surround_v2.env(
    obs_type="ram",
    full_action_space=True,
    max_cycles=1000,
    auto_rom_install_path=ROM_PATH,
    render_mode="rgb_array",
)
env = RecordVideo(env, video_folder=VIDEO_FOLDER, name_prefix="surround-video")

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    # this is where you would insert your policy
    action = None if termination or truncation else env.action_space(agent).sample()
    env.step(action)
env.close()
