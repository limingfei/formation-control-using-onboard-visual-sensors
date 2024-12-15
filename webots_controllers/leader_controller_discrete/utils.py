import numpy as np
from scipy.spatial.transform import Rotation as R
def obtain_relative_pose(follower,leader):

    follower_positon = follower.getPosition()
    leader_positon = leader.getPosition()
    follower_orien = np.array(follower.getOrientation())
    follower_orien = np.reshape(follower_orien,(3,3))
    leader_orien = np.array(leader.getOrientation())
    leader_orien = np.reshape(leader_orien,(3,3))
    position_a = follower_positon
    rotA = follower_orien
    position_b = leader_positon
    rotB = leader_orien

    # 机器人A的位置和旋转矩阵
    x1, y1, z1 = position_a # 例子中的值
    rotA = np.array(rotA)
    rotB = np.array(rotB)
 

    # 机器人B的位置和旋转矩阵
    x2, y2, z2 = position_b  # 例子中的值
  

    # 计算位置向量的差
    delta_position = np.array([x2 - x1, y2 - y1, z2 - z1])

    # 使用A的旋转矩阵的转置将delta_position旋转到A的局部坐标系
    local_position = rotA.T.dot(delta_position)

    # 计算局部旋转矩阵
    local_orientation = rotA.T.dot(rotB)
    r = R.from_matrix(local_orientation)
    el = r.as_euler('xyz')

    # 输出结果
    return local_position,el[-1]
def set_mecanum_wheel_speeds(vx, vy, omega,wheels):
    # 计算每个轮子的速度
    wheel_radius = 0.254/2 # 轮子半径，需要根据实际模型调整
    wheel_distance = 0.27+0.27  # 轮子距离中心的距离，同样需要调整

    fl_speed = (vx - vy - (wheel_distance * omega)) / wheel_radius
    fr_speed = (vx + vy + (wheel_distance * omega)) / wheel_radius
    bl_speed = (vx + vy - (wheel_distance * omega)) / wheel_radius
    br_speed = (vx - vy + (wheel_distance * omega)) / wheel_radius

    # 设置轮子速度
    # print(fl_speed,fr_speed,bl_speed,br_speed)
    for wheel,speed in zip(wheels,(fl_speed,fr_speed,bl_speed,br_speed)):
        wheel.setPosition(float('inf'))
        wheel.setVelocity(speed)