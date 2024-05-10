from cr5_env.cr5_reach_visual_mlp import CR5ReachVisualEnv
import torch
from colorama import Fore, init

init()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = CR5ReachVisualEnv(is_good_view=True, is_render=True)
obs = env.reset()

ac = torch.load('logs/ppo-cr5_mesh-reach_2023-03-23/ppo-cr5_mesh-reach_10-49-33/pyt_save/model.pt')

actions = ac.act(torch.as_tensor(obs, dtype=torch.float32).to(device))

sum_reward = 0
success_times = 0
for i in range(50):
    obs = env.reset()
    for step in range(env.max_steps_one_episode):
        actions = ac.act(torch.as_tensor(obs, dtype=torch.float32).to(device))
        obs, reward, done, info = env.step(actions)
        sum_reward += reward
        if reward == 1:
            success_times += 1
        if done:
            break

print(Fore.GREEN + "sum_reward={}".format(sum_reward))
print(Fore.GREEN + "success rate={}".format(success_times / 50))
