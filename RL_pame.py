import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import pandas as pd
from pame_2 import PAME


# --- 强化学习智能体和网络 ---
class Policy(nn.Module):
    """
    一个简单的策略网络，同时输出动作概率和状态价值。
    """

    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)
        state_value = self.critic(state)
        return dist, state_value


def train(optimizer_name, policy_network, env):
    """
    使用指定的优化器训练智能体。
    """
    print(f"--- 开始训练: {optimizer_name} ---")

    # 根据名称选择优化器
    if optimizer_name == 'PAME':
        optimizer = PAME(policy_network.parameters(), lr=0.001, gamma=0.999)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(policy_network.parameters(), lr=0.001)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(policy_network.parameters(), lr=0.001)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(policy_network.parameters(), lr=0.001, momentum=0.9)
    elif optimizer_name == 'RAdam':
        optimizer = optim.RAdam(policy_network.parameters(), lr=0.001)
    else:
        raise ValueError(f"未知的优化器: {optimizer_name}")

    max_episodes = 2000
    gamma_discount = 0.99
    all_rewards = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        log_probs, values, rewards = [], [], []

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            dist, state_value = policy_network(state_tensor)

            action = dist.sample()
            next_state, reward, terminated, truncated, _ = env.step(action.item())

            log_probs.append(dist.log_prob(action))
            values.append(state_value)
            rewards.append(reward)
            state = next_state

            if terminated or truncated:
                break

        returns = []
        discounted_reward = 0
        for r in rewards[::-1]:
            discounted_reward = r + gamma_discount * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze()

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = nn.functional.mse_loss(values, returns)

        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        all_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{max_episodes}, 总奖励: {total_reward:.2f}")

    print(f"--- 训练完成: {optimizer_name} ---")
    return all_rewards


# --- 主程序和结果保存 ---
def save_results_to_csv(results, filename="rl_performance_results.csv"):
    """
    将所有优化器的结果保存到一个CSV文件中。
    """
    # 使用 pandas 创建一个 DataFrame
    df = pd.DataFrame(results)
    # 添加一个代表回合数的列
    df.index.name = 'Episode'
    df.index = df.index + 1

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(filename)
    print(f"\n结果已成功保存到文件: {filename}")


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    optimizers_to_test = ['PAME', 'Adam', 'AdamW', 'RAdam', 'SGD']
    all_results = {}

    for optimizer_name in optimizers_to_test:
        # 为每个优化器重新初始化网络，保证公平对比
        policy_net = Policy(state_dim, action_dim)
        rewards = train(optimizer_name, policy_net, env)
        all_results[optimizer_name] = rewards

    env.close()

    # 保存结果到CSV文件
    save_results_to_csv(all_results)
