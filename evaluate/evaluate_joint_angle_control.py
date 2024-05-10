from cr5_env.cr5_joint_angle_control import CR5AvoidVisualEnv
import torch
from colorama import Fore, init

init()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = CR5AvoidVisualEnv(is_good_view=False, is_render=False)
obs = env.reset()

ac = torch.load('../logs/joint-angle-avoid_2023-06-20/joint-angle-avoid_20-17-11'
                '/pyt_save/model.pt')

actions = ac.act(torch.as_tensor(obs, dtype=torch.float32).to(device))

sum_reward = 0
success_times = 0
total_num = 1
for i in range(total_num):
    obs = env.reset()
    for step in range(env.max_steps_one_episode):
        actions = ac.act(torch.as_tensor(obs, dtype=torch.float32).to(device))
        obs, reward, done, info = env.step(actions)
        print(env.newJointAngles)
        sum_reward += reward
        if env.reach_check and env.obs_distance > 0.05:
            success_times += 1
        if done:
            break

print(Fore.GREEN + "mean_reward={}".format(sum_reward / total_num))
print(Fore.GREEN + "success rate={}".format(success_times / total_num))
