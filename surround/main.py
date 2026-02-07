import os

from pettingzoo.atari import surround_v2

ROM_PATH = os.path.expanduser("~/.local/share/AutoROM/roms")
env = surround_v2.env(
    obs_type="ram",
    full_action_space=True,
    max_cycles=1000,
    auto_rom_install_path=ROM_PATH,
)

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()
    env.step(action)
env.close()