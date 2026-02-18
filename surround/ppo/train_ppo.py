import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from surround.conf import constants
from surround.dqn.train_dqn import get_state_from_observation


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, action_dim), nn.Softmax(dim=-1)
        )

        # Critic
        self.critic = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)
        return torch.distributions.Categorical(probs), value


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, epochs=10):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.mse_loss = nn.MSELoss()

    def update(self, memory):
        states = torch.tensor(np.array(memory["states"]), dtype=torch.float32)
        actions = torch.tensor(memory["actions"])
        old_log_probs = torch.stack(memory["log_probs"]).detach()
        rewards = memory["rewards"]
        is_terminals = memory["is_terminals"]

        # Calculate rewards-to-go
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32)
        # normalize for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # and run
        for episode_index in trange(self.epochs):
            dist, state_values = self.policy(states)
            state_values = state_values.squeeze()
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            advantages = returns - state_values.detach()
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.mse_loss(state_values, returns)
                - 0.01 * entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def main():
    gym.register_envs(ale_py)
    env = gym.make(
        "ALE/Surround-v5",
        obs_type="grayscale",
        full_action_space=False,
        difficulty=constants.DIFFICULTY,
        mode=constants.MODE,
        frameskip=constants.DQN_FRAME_SKIP,
    )

    state_dim = 7  # TODO(bjafek) state_tuple from DQN
    agent = PPO(state_dim, constants.N_ACTIONS)
    max_episodes = 400
    update_timestep = 1000
    timestep = 0

    for episode_index in trange(max_episodes):
        observation, _ = env.reset()
        state = np.array(get_state_from_observation(observation, 1))
        memory = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "is_terminals": [],
        }
        ep_reward = 0

        for cycle_num in trange(constants.MAX_CYCLES):
            timestep += 1

            # 1. select action
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                dist, _ = agent.policy(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # 2. step environment
            next_obs, reward, terminated, truncated, _info = env.step(action.item())
            last_action = memory["actions"][-1] if memory["actions"] else 1
            next_state = np.array(get_state_from_observation(next_obs, last_action=last_action))
            done = terminated or truncated or abs(reward) == 1

            # 3. save to memory
            memory["states"].append(state)
            memory["actions"].append(action.item())
            memory["log_probs"].append(log_prob)
            memory["rewards"].append(reward)
            memory["is_terminals"].append(done)

            state = next_state
            ep_reward += reward

            # 4. update PPO agent
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory = {k: [] for k in memory}
                timestep = 0

            if done:
                break

        if episode_index % 20 == 0:
            print(f"Episode {episode_index} \t Last Reward: {ep_reward}")

    print("Training complete!")
    env.close()


if __name__ == "__main__":
    main()
