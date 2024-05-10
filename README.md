# 基于PPO算法机械臂轨迹规划仿真
## 1."cr5_env"文件夹中程序包含仿真场景设计，包括模型初始化，机械臂状态设计，奖励函数设计，目标物体和障碍物设计等。
### 1.1 “cr5_gripper_obstacle_visual_mlp.py”--此程序设计了带夹爪模型的CR5机械臂避障场景，以末端位置x,y,z为状态，通过MLP训练动作策略；
### 1.2 “cr5_joint_angle_control.py”--此程序以机械臂关节值为状态设计仿真场景；

## 2.“train"与"evaluate"文件夹分别为训练和评估程序，需要保证程序中导入的env仿真环境一致，直接运行即可。

## 3. “utils/handmodel.py"为手臂障碍物模型。