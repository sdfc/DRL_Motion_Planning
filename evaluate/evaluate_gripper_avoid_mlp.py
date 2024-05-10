from cr5_env.cr5_gripper_obstacle_visual_mlp import CR5AvoidVisualEnv
import torch
from colorama import Fore, init

init()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = CR5AvoidVisualEnv(is_good_view=True, is_render=True)
obs = env.reset()

ac = torch.load("../logs/ppo-gripper-avoid-with-ou-randomGoal_2023-05-02/ppo-gripper-avoid-with-ou-randomGoal_10-04-59/pyt_save/model.pt")

actions = ac.act(torch.as_tensor(obs, dtype=torch.float32).to(device))

sum_reward = 0
success_times = 0
total_num = 100

print("+++++++++++++++++++++++++++++")
print("+++++++++++++++++++++++++++++")
for i in range(total_num):
    obs = env.reset()
    for step in range(env.max_steps_one_episode):
        actions = ac.act(torch.as_tensor(obs, dtype=torch.float32).to(device))
        print(actions)

        obs, reward, done, info = env.step(actions)
        sum_reward += reward
        if env.reach_check and env.obs_distance > 0:
            success_times += 1
        if done:
            break
    print("+++++++++++++++++++++++++++++")
    print("+++++++++++++++++++++++++++++")

print(Fore.GREEN + "mean_reward={}".format(sum_reward / total_num))
print(Fore.GREEN + "success rate={}".format(success_times / total_num))
