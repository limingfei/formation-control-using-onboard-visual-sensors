"""leader_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from utils import set_mecanum_wheel_speeds
import ctypes
import numpy as np
np.random.seed(11)
pub_leader_rate = 50
pub_num = 0

robot = Robot()
lib = ctypes.CDLL('/home/lmf/文档/youbot/libraries/youbot_control/src/libbase.so')
lib.wb_robot_init()
lib.base_init()
lib.set_speed.argtypes = [ctypes.c_double]
lib.set_max_speed.argtypes = [ctypes.c_double]
lib.set_max_omega.argtypes = [ctypes.c_double]
lib.set_speed_increment.argtypes = [ctypes.c_double]
lib.base_move.argtypes = [ctypes.c_double,ctypes.c_double,ctypes.c_double]
lib.set_max_speed(0.2)
lib.set_max_omega(0.1)
lib.set_speed_increment(0.03)



timestep = int(robot.getBasicTimeStep())
# wheels = []
# for wheel_name in ['leader_fl_wheel_motor','leader_fr_wheel_motor','leader_rl_wheel_motor','leader_rr_wheel_motor']:
#     wheel = robot.getDevice(wheel_name)
#     wheel.setPosition(float('inf'))
#     wheel.setVelocity(0.0)
#     wheels.append(wheel)

while robot.step(timestep) != -1:
    # print('-------------',np.random)
    # print('********************',type(np.random))
    # lib.base_forwards_increment()
    lib.base_strafe_left_increment()


