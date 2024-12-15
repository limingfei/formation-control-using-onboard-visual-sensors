"""leader_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from utils import set_mecanum_wheel_speeds
import ctypes
import numpy as np
np.random.seed(10)
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
x = np.random.uniform(-0.2,0.2)
y = np.random.uniform(-0.2,0.2)
z = np.random.uniform(-0.1,0.1)
behavior  = np.random.choice([0,1,2,3,4,5,6])

while robot.step(timestep) != -1:
    # print('-------------',np.random)
    # print('********************',type(np.random))

    if pub_num >= pub_leader_rate:
        behavior  = np.random.choice([0,1,2,3,4,5,6])
        # x = np.random.uniform(-0.2,0.2)
        # y = np.random.uniform(-0.2,0.2)
        # z = np.random.uniform(-0.1,0.1)
        # x = np.random
        # if behavior == 0:
        #     y,z = 0,0
        # elif behavior == 1:
        #     x,z = 0,0
        # elif behavior == 2:
        #     x,y = 0,0
        # elif behavior == 3:
        #     x = 0
        # elif behavior == 4:
        #     y = 0
        # elif behavior == 5:
        #     z = 0
        # else:
        #    continue
        # print('x is :',x)
        # print('y is :',y)
        # print('z is :',z)

        # x = np.random.normal(0.2,0.03)
        # x = np.random.choice((x,-x))
        # y = np.random.normal(0.2,0.03)
        # y = np.random.choice((y,-y))
        # z = np.random.normal(0.1,0.02)
        # z = np.random.choice((z,-z))
        pub_num = 0
    # set_mecanum_wheel_speeds(x,y,z,wheels)
    # lib.base_move(x,y,z)

    if behavior == 0:
        pass
    elif behavior == 1:
        lib.base_forwards_increment()
    elif behavior == 2:
        lib.base_backwards_increment()
    elif behavior == 3:
        lib.base_turn_left_increment()
    elif behavior == 4:
        lib.base_turn_right_increment()
    elif behavior == 5:
        lib.base_strafe_left_increment()
    elif behavior == 6:
        lib.base_strafe_right_increment()
    pub_num += 1


