import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class HandModel:
    def __init__(self, sphere_center, sphere_radius=0.2, cylinder_height=0.8, cylinder_radius=0.1, theta=60,
                 rotation_axis='y'):
        self.cylinder = None
        self.sphere = None
        self.center = sphere_center  # 球心
        self.radius = sphere_radius  # 球体半径
        self.height = cylinder_height  # 圆柱体高度
        self.r = cylinder_radius  # 圆柱底面半径
        self.theta = theta  # 圆柱倾斜角度
        self.rot_axis = rotation_axis

        self.u = 10
        self.v = 10
        self.h = 20
        self.t = 10
        self.points_num = int(self.u * self.v + self.h * self.t)

    def create_model(self):
        # 生成球体坐标
        u = np.linspace(0, 2 * np.pi, self.u)
        v = np.linspace(0, np.pi, self.v)
        x = self.center[0] + self.radius * np.outer(np.cos(u), np.sin(v))
        y = self.center[1] + self.radius * np.outer(np.sin(u), np.sin(v))
        z = self.center[2] + self.radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # 生成圆柱体坐标
        h = np.linspace(0, self.height, self.h)
        theta_ = np.linspace(0, 2 * np.pi, self.t)
        theta_, h = np.meshgrid(theta_, h)
        x_cylinder = self.center[0] + self.r * np.cos(theta_)
        y_cylinder = self.center[1] + self.r * np.sin(theta_)
        z_cylinder = self.center[2] + h

        if self.rot_axis == 'x':
            # 将圆柱体绕x轴旋转theta度
            theta = np.radians(self.theta)
            x_cylinder, z_cylinder = (x_cylinder - self.center[0], z_cylinder - self.center[2])
            x_cylinder, z_cylinder = (x_cylinder * np.cos(theta) - z_cylinder * np.sin(theta),
                                      x_cylinder * np.sin(theta) + z_cylinder * np.cos(theta))
            x_cylinder, z_cylinder = (x_cylinder + self.center[0], z_cylinder + self.center[2])
            # 将圆柱体移动到球体表面
            # x_cylinder -= (self.radius-0.02) * np.sin(theta)
            # z_cylinder += (self.radius-0.02) * np.cos(theta)
        elif self.rot_axis == 'y':
            # 将圆柱体绕y轴旋转theta度
            theta = np.radians(self.theta)
            z_cylinder, y_cylinder = (z_cylinder - self.center[2], y_cylinder - self.center[1])
            z_cylinder, y_cylinder = (z_cylinder * np.cos(theta) - y_cylinder * np.sin(theta),
                                      z_cylinder * np.sin(theta) + y_cylinder * np.cos(theta))
            z_cylinder, y_cylinder = (z_cylinder + self.center[2], y_cylinder + self.center[1])
            # 将圆柱体移动到球体表面
            # y_cylinder += (self.radius-0.02) * np.sin(theta)
            # z_cylinder += (self.radius-0.02) * np.cos(theta)
        else:
            raise ValueError("旋转轴错误")

        # 用于画图
        self.sphere = np.array([x, y, z])
        self.cylinder = np.array([x_cylinder, y_cylinder, z_cylinder])

        # 将圆柱和球体的所有点合并在一个列表中
        points = np.concatenate([np.array([x.flatten(), y.flatten(), z.flatten()]).T,
                                 np.array([x_cylinder.flatten(), y_cylinder.flatten(), z_cylinder.flatten()]).T],
                                axis=0)

        return points

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(self.sphere[0], self.sphere[1], self.sphere[2], color="gray")
        ax.plot_surface(self.cylinder[0], self.cylinder[1], self.cylinder[2], color="b")
        plt.show()


def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)


def min_distance(point, point_set):
    min_dist = float('inf')
    for p in point_set:
        dist = distance(point, p)
        if dist < min_dist:
            min_dist = dist
    return min_dist


if __name__ == '__main__':
    center = np.array([1, 2, 3])
    hand_model = HandModel(sphere_center=center, rotation_axis='y')
    points_set = hand_model.create_model()
    print(points_set.shape)
    hand_model.plot()
