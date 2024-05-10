import pybullet as p
import pybullet_data  # pybullet自带的一些模型
import os
from PIL import Image

urdf_root_path = pybullet_data.getDataPath()
p.connect(p.GUI)  # 连接到仿真环境，p.DIREACT是不显示仿真界面,p.GUI则为显示仿真界面
p.setGravity(0, 0, -10)  # 设定重力
p.resetSimulation()  # 重置仿真环境
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 添加pybullet的额外数据地址，使程序可以直接调用到内部的一些模型
p.loadURDF(os.path.join(urdf_root_path, "plane.urdf"),
                   basePosition=[0, 0, -0.65])  # 加载外部平台模型
tabel = p.loadURDF(os.path.join(urdf_root_path, "table/table.urdf"),
                               basePosition=[0.5, -0.25, -0.65])
# # cr5_id = p.loadURDF("../URDF_model/urdf/cr5_robot.urdf", useFixedBase=True)
# cr5_id = p.loadURDF("../URDF_model/urdf/cr5_robot_gripper.urdf", useFixedBase=True)
cr5_id = p.loadURDF(os.path.join(urdf_root_path, "cr5/cr5_robot_gripper.urdf"), useFixedBase=True)
init_joint_positions = [0, 0.81, -2.64, 1.57, 1.53, 0]
# for i in range(6):
#     p.resetJointState(
#         bodyUniqueId=cr5_id,
#         jointIndex=i,
#         targetValue=init_joint_positions[i],
#     )
# 设置相机位置和朝向
view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0.2],
                                                  distance=2.5,
                                                  yaw=70,
                                                  pitch=-45,
                                                  roll=0,
                                                  upAxisIndex=2)

# 获取当前画面像素数据
proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                           aspect=1.0,
                                           nearVal=0.01,
                                           farVal=100.0)
width, height, rgb_pixels, depth_pixels, segmentation_mask = p.getCameraImage(width=320,
                                                                               height=240,
                                                                               viewMatrix=view_matrix,
                                                                               projectionMatrix=proj_matrix)

# 将像素数据转换为图像，并保存到本地文件
rgb_array = Image.fromarray(rgb_pixels)
# rgb_array.save("test.png")

while 1:
    p.stepSimulation()
    