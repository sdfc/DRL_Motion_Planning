import pybullet as p
import pybullet_data
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from math import sqrt
import random
import time
import math
import cv2
import torch
import os
from colorama import Fore
from utils.handmodel import *
from PIL import Image


class CR5AvoidVisualEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    kImageSize = {'width': 96, 'height': 96}
    kFinalImageSize = {'width': 84, 'height': 84}

    def __init__(self, is_render=False, is_good_view=False, max_steps=4000):

        self.newJointAngles = None
        self.arm_model = None
        self.images = None
        self.arm_center = None
        self.arm_points = None
        self.arm_id = None
        self.terminated = None
        self.max_steps_one_episode = max_steps
        self.num_joints = None
        self.obstacle_obj_id = None
        self.goal_obj_id = None
        self.cr5_id = None
        self.is_render = is_render
        self.is_good_view = is_good_view

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.gripper_length = 0.16

        self.x_low_obs = -0.85
        self.x_high_obs = -0.15
        self.y_low_obs = -0.27
        self.y_high_obs = 0.27
        self.z_low_obs = 0
        self.z_high_obs = 0.55

        self.x_low_obstacle = -0.48
        self.x_high_obstacle = -0.35
        self.y_low_obstacle = -0.20
        self.y_high_obstacle = 0.07
        self.z_low_obstacle = 0.07
        self.z_high_obstacle = 0.24

        self.j1_low_action = -5
        self.j1_high_action = 5
        self.j2_low_action = -5
        self.j2_high_action = 5
        self.j3_low_action = -5
        self.j3_high_action = 5
        self.j4_low_action = -2
        self.j4_high_action = 2
        self.j5_low_action = -5
        self.j5_high_action = 5
        self.j6_low_action = -5
        self.j6_high_action = 5

        self.j1_low_action = 0
        self.j1_high_action =0
        self.j2_low_action = 0
        self.j2_high_action =0
        self.j3_low_action = 0
        self.j3_high_action =0
        self.j4_low_action = 0
        self.j4_high_action =0
        self.j5_low_action = 0
        self.j5_high_action =0
        self.j6_low_action = 0
        self.j6_high_action =0

        self.step_counter = 0
        self.end_effector_index = 6
        self.gripper_index = 7

        self.target_pose = np.array([[-0.82, -0.12, 0.07],
                                     [-0.72, -0.12, 0.10],
                                     [-0.82, -0.22, 0.10],
                                     [-0.72, -0.22, 0.07]])

        self.urdf_root_path = pybullet_data.getDataPath()
        # lower limits for null space
        self.lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # rest-poses for null space
        self.rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficients
        self.joint_damping = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]

        self.init_joint_positions = [0, -35, 115, 11, -90, 40, 0, 0]  # 0,28.65,-160.43,90,87.66,0

        # self.orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])
        self.orientation = np.array([math.pi, 0., math.pi / 4.])

        p.configureDebugVisualizer(lightPosition=[5, 0, 5])
        # p.resetDebugVisualizerCamera(1, 90, -30, [1.5, 0, 1])
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.action_space = spaces.Box(
            low=np.array([
                self.j1_low_action,
                self.j2_low_action,
                self.j3_low_action,
                self.j4_low_action,
                self.j5_low_action,
                self.j6_low_action
            ]),
            high=np.array([
                self.j1_high_action,
                self.j2_high_action,
                self.j3_high_action,
                self.j4_high_action,
                self.j5_high_action,
                self.j6_high_action
            ]),
            dtype=np.float32)

        # self.observation_space = spaces.Box(low=0, high=1,
        #                                     shape=(1, self.kFinalImageSize['width'], self.kFinalImageSize['height']))

        self.observation_space = spaces.Box(
            low=np.array(  # [0] * self.arm_model.points_num * 3 +
                [
                    self.x_low_obs, self.y_low_obs, self.z_low_obs,
                    self.x_low_obs, self.y_low_obs, self.z_low_obs,
                    self.x_low_obs, self.y_low_obs, self.z_low_obs
                ]),
            high=np.array(  # [1] * self.arm_model.points_num * 3 +
                [
                    self.x_high_obs, self.y_high_obs, self.z_high_obs,
                    self.x_high_obs, self.y_high_obs, self.z_high_obs,
                    self.x_high_obs, self.y_high_obs, self.z_high_obs
                ]),
            dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_counter = 0

        p.resetSimulation()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated = False
        p.setGravity(0, 0, -10)

        '''生成边界线'''
        # self.generate_region()

        '''模型导入'''
        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"),
                   basePosition=[0, 0, -0.65])

        self.cr5_id = p.loadURDF(os.path.join(self.urdf_root_path, "cr5/cr5_robot_gripper.urdf"),
                                 basePosition=[0, 0, 0],
                                 useFixedBase=True)

        table_uid = p.loadURDF(os.path.join(self.urdf_root_path, "table/table.urdf"), basePosition=[-0.5, 0, -0.63])
        p.changeVisualShape(table_uid, -1, rgbaColor=[1, 1, 1, 1])

        # 目标点模型
        self.goal_obj_id = p.loadURDF(os.path.join(self.urdf_root_path,
                                                   "cube_small.urdf"),  # random_urdfs/120/120.urdf
                                      basePosition=self.target_pose[np.random.choice(self.target_pose.shape[0])]
                                      )

        # 创建手臂模型
        # self.arm_center = [-0.5, 0, 0.15]
        self.arm_center = [random.uniform(self.x_low_obstacle, self.x_high_obstacle),
                           random.uniform(self.y_low_obstacle, self.y_high_obstacle),
                           random.uniform(self.z_low_obstacle, self.z_high_obstacle)]
        theta = random.uniform(40, 75)
        self.arm_model = HandModel(sphere_center=self.arm_center, rotation_axis='y', theta=theta,
                                   sphere_radius=0.05, cylinder_radius=0.03, cylinder_height=0.25)
        self.arm_points = self.arm_model.create_model()

        r, c = self.arm_points.shape
        colors = np.ones((r, c)) * 0.5
        size = 5
        self.arm_id = p.addUserDebugPoints(self.arm_points, colors, size)

        p.changeDynamics(self.goal_obj_id, -1, mass=0)

        self.num_joints = p.getNumJoints(self.cr5_id)

        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.cr5_id,
                jointIndex=i,
                targetValue=np.radians(self.init_joint_positions[i]),
            )

        self.robot_pos_obs = p.getLinkState(self.cr5_id,
                                            self.num_joints - 1)[4]

        p.stepSimulation()
        p.resetJointState(self.cr5_id, 7, -0.3)

        # self.object_pos = p.getBasePositionAndOrientation(self.goal_obj_id)[0]
        # self.obstacle_pos = p.getBasePositionAndOrientation(self.obstacle_obj_id)[0]
        # self.observation_pose = self.obstacle_pos + self.object_pos

        return self._resolve_obs_return()

    def step(self, action):

        dv = 0.5
        d_j1 = action[0] * dv
        d_j2 = action[1] * dv
        d_j3 = action[2] * dv
        d_j4 = action[3] * dv
        d_j5 = action[4] * dv
        d_j6 = action[5] * dv

        action_step = [d_j1, d_j2, d_j3, d_j4, d_j5, d_j6]

        # for i, rad_s in enumerate(action_step):
        #     p.setJointMotorControl2(self.cr5_id, i, p.VELOCITY_CONTROL, rad_s)

        self.newJointAngles = []
        for j in range(6):
            jointState = p.getJointState(self.cr5_id, j)
            jointAngle = np.degrees(jointState[0])  # rad -> deg
            jointAngle += action_step[j]
            self.newJointAngles.append(jointAngle)

        for i in range(self.end_effector_index):
            p.resetJointState(
                bodyUniqueId=self.cr5_id,
                jointIndex=i,
                targetValue=np.radians(self.newJointAngles[i])  # deg -> rad
            )

        # current_pos = p.getLinkState(self.cr5_id, self.end_effector_index)[4]
        # self.new_robot_pos = [
        #     current_pos[0] + dx, current_pos[1] + dy,
        #     current_pos[2] + dz
        # ]
        # self.robot_joint_positions = p.calculateInverseKinematics(
        #     bodyUniqueId=self.cr5_id,
        #     endEffectorLinkIndex=self.num_joints - 1,
        #     targetPosition=[
        #         self.new_robot_pos[0], self.new_robot_pos[1],
        #         self.new_robot_pos[2]
        #     ],
        #     targetOrientation=self.orientation,
        #     jointDamping=self.joint_damping,
        # )
        #
        # for i in range(self.end_effector_index):
        #     p.resetJointState(
        #         bodyUniqueId=self.cr5_id,
        #         jointIndex=i,
        #         targetValue=self.robot_joint_positions[i],
        #     )
        p.stepSimulation()

        # 在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        if self.is_good_view:
            time.sleep(0.05)

        self.step_counter += 1

        return self._reward()

    def _reward(self):

        # 一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_state = np.array(
            p.getLinkState(self.cr5_id, self.end_effector_index)[4]).astype(np.float32)

        pos, rot = p.getLinkState(self.cr5_id, self.end_effector_index)[:2]
        euler = np.array(p.getEulerFromQuaternion(rot))

        self.robot_state[2] -= self.gripper_length

        obj_x = self.robot_state[0]
        obj_y = self.robot_state[1]
        obj_z = self.robot_state[2]

        self.goal_obj_state = np.array(
            p.getBasePositionAndOrientation(self.goal_obj_id)[0]).astype(
            np.float32)

        self.obstacle_obj_state = np.array(self.arm_center).astype(np.float32)

        # 用机械臂末端和物体的距离作为奖励函数的依据
        self.obj_distance = distance(self.robot_state, self.goal_obj_state)

        # 机械臂末端和手臂模型点集最短距离
        self.obs_distance = min_distance(self.robot_state, self.arm_points)

        # 机械臂末端与y_min边界的距离，意味着不可干扰人类工作空间
        self.boundary_distance_y = sqrt((obj_y - self.y_low_obs) ** 2)
        # 机械臂末端与z_min边界的距离，不允许机械臂从下方绕过障碍物
        self.boundary_distance_z = sqrt((obj_z - self.z_low_obs) ** 2)

        # 机械臂碰到障碍物，视为done，给予一定的惩罚
        # collision_points = p.getContactPoints(self.cr5_id, self.collision_id)
        # collision_check = bool(len(collision_points) > 0)

        # 如果机械臂末端超过了obs的空间，也视为done，而且会给予一定的惩罚
        # boundary_check = bool(obj_x < self.x_low_obs or obj_x > self.x_high_obs
        #                       or obj_y < self.y_low_obs or obj_y > self.y_high_obs
        #                       or obj_z < self.z_low_obs or obj_z > self.z_high_obs)
        boundary_check = bool(obj_x > self.x_high_obs or obj_z < self.z_low_obs)

        # 当前姿态与设定姿态的差值作为奖励函数一部分
        self.ori_diff = np.sum(abs(self.orientation - euler))

        # 判定任务完成指标，机械臂到达目标点正上方区域
        self.reach_check = bool(abs(self.goal_obj_state[0] - obj_x) < 0.06
                                and abs(self.goal_obj_state[1] - obj_y) < 0.06
                                and obj_z - self.goal_obj_state[2] < 0.16)

        '''%%%此处使用离散值作为奖励%%%'''
        # if boundary_check:
        #     reward = -0.1
        #     self.terminated = True
        # # 如果机械臂一直无所事事，在最大步数还不能接触到物体，也需要给一定的惩罚
        # elif self.step_counter > self.max_steps_one_episode:
        #     reward = -0.1
        #     self.terminated = True
        # elif self.obs_distance < 0.05:
        #     reward = -0.01
        #     self.terminated = True
        # elif self.obj_distance < 0.12:
        #     reward = 1
        #     self.terminated = True
        # else:
        #     reward = 0
        #     self.terminated = False
        '''%%%===================%%%'''

        '''%%%此处使用距离函数作为奖励%%%'''
        delta = 0.15
        m = 7
        n = 6
        d_o = 0.6
        k = 0.1
        c1 = 15.0
        c2 = 0.85
        c3 = 0.06

        R_T = np.where(np.abs(self.obj_distance) < delta, 0.5 * (self.obj_distance ** 2),
                       delta * np.abs(self.obj_distance) - 0.5 * (delta ** 2))
        R_O = pow(d_o / (self.obs_distance + d_o), m)
        R_Bz = pow(k / self.boundary_distance_z, n)

        reward_ = -(c1 * R_T + c2 * R_O + c3 * R_Bz)

        if self.reach_check:
            self.terminated = True
            reward_ = reward_ + abs(10 * reward_)
            if self.ori_diff < 0.2:
                reward_ = reward_ + abs(10 * reward_)
        elif boundary_check:
            self.terminated = True
            reward_ = reward_ - abs(100 * reward_)
        elif self.step_counter > self.max_steps_one_episode:
            self.terminated = True
            reward_ = reward_ - abs(10 * reward_)
        else:
            self.terminated = False
        '''%%%===================%%%'''

        info_ = ('distance: ' + str(self.obj_distance))

        return self._resolve_obs_return(), reward_, self.terminated, info_

    def _resolve_obs_return(self):
        obstacle_position = list(self.arm_center)
        # obstacle_position = list(self.arm_points.flatten())
        goal_position = list(p.getBasePositionAndOrientation(self.goal_obj_id)[0])
        robot_end_effector_position = list(p.getLinkState(self.cr5_id, self.end_effector_index)[4])

        return np.array(obstacle_position + goal_position + robot_end_effector_position).astype(np.float32)

    def close(self):
        p.disconnect()

    def generate_region(self):
        """边界区域"""
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        '''障碍物生成区域'''
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obstacle, self.y_low_obstacle, self.z_low_obstacle],
            lineToXYZ=[self.x_low_obstacle, self.y_low_obstacle, self.z_high_obstacle])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obstacle, self.y_high_obstacle, self.z_low_obstacle],
            lineToXYZ=[self.x_low_obstacle, self.y_high_obstacle, self.z_high_obstacle])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obstacle, self.y_low_obstacle, self.z_low_obstacle],
            lineToXYZ=[self.x_high_obstacle, self.y_low_obstacle, self.z_high_obstacle])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obstacle, self.y_high_obstacle, self.z_low_obstacle],
            lineToXYZ=[self.x_high_obstacle, self.y_high_obstacle, self.z_high_obstacle])

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obstacle, self.y_low_obstacle, self.z_high_obstacle],
            lineToXYZ=[self.x_high_obstacle, self.y_low_obstacle, self.z_high_obstacle])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obstacle, self.y_high_obstacle, self.z_high_obstacle],
            lineToXYZ=[self.x_high_obstacle, self.y_high_obstacle, self.z_high_obstacle])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obstacle, self.y_low_obstacle, self.z_high_obstacle],
            lineToXYZ=[self.x_low_obstacle, self.y_high_obstacle, self.z_high_obstacle])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obstacle, self.y_low_obstacle, self.z_high_obstacle],
            lineToXYZ=[self.x_high_obstacle, self.y_high_obstacle, self.z_high_obstacle])

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obstacle, self.y_low_obstacle, self.z_low_obstacle],
            lineToXYZ=[self.x_high_obstacle, self.y_low_obstacle, self.z_low_obstacle])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obstacle, self.y_low_obstacle, self.z_low_obstacle],
            lineToXYZ=[self.x_low_obstacle, self.y_high_obstacle, self.z_low_obstacle])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obstacle, self.y_high_obstacle, self.z_low_obstacle],
            lineToXYZ=[self.x_high_obstacle, self.y_low_obstacle, self.z_low_obstacle])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obstacle, self.y_high_obstacle, self.z_low_obstacle],
            lineToXYZ=[self.x_low_obstacle, self.y_high_obstacle, self.z_low_obstacle])


if __name__ == '__main__':
    # 这一部分是做baseline，即让机械臂随机选择动作，看看能够得到的分数

    env = CR5AvoidVisualEnv(is_render=True, is_good_view=True)
    print(env.observation_space.shape)
    print(env.action_space.shape)
    for _ in range(1000):
        action = env.action_space.sample()
        print('action={}'.format(action))
        env.step(action)
        obs, reward, done, info = env.step(action)
        print('obs={} | reward={} | done={} | info={}'.format(obs, reward, done, info))
        print("obstacle distance: {}".format(env.obs_distance))
        print("object distance: {}".format(env.obj_distance))
        print("orientation diff: {}".format(env.ori_diff))
        # print("R_T: {} |R_O: {} |R_B: {} |reward: {}".format(R_T, R_O, R_B, reward_))
        if done:
            env.reset()

    state = env.reset()
    print(state.shape)
