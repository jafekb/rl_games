from pathlib import Path

from pettingzoo.atari import surround_v2

ROM_PATH = str(Path("~/.local/share/AutoROM/roms").expanduser())
env = surround_v2.env(
    obs_type="ram",
    full_action_space=True,
    max_cycles=1000,
    auto_rom_install_path=ROM_PATH,
)

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    # this is where you would insert your policy
    action = None if termination or truncation else env.action_space(agent).sample()
    env.step(action)
env.close()
