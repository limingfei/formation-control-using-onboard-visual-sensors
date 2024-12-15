from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
import cv2
import numpy as np
import yaml
from utils import obtain_relative_pose,set_mecanum_wheel_speeds,random_occlusion,random_bright,random_contrast,random_gaussian_blur,add_snow
from rich.console import Console
import os
import ctypes
class WebotsEnv(RobotSupervisorEnv):
    def __init__(self,parms_path,eval=False):
        super().__init__()
        
        with open(parms_path,'r') as file:
            self.parms = yaml.safe_load(file)
        filename, _ = os.path.splitext(os.path.basename(parms_path))
        # assert self.parms['config_file'] == filename
        print(self.parms)
        self.observation_type = self.parms['observation_type']
        if self.observation_type == 'color_image':
            self.observation_space = np.array([self.parms['observation_shape']['channel'],self.parms['observation_shape']['width'],self.parms['observation_shape']['height']])
        elif self.observation_type == 'low_features':
            self.observation_space = 6
        self.action_space = int(self.parms['action_shape'])
        self.robot = self.getSelf()        
        # self.wheels = []
        # for wheel_name in self.parms['wheel_names']:
        #     wheel = self.getDevice(wheel_name)
        #     wheel.setPosition(float('inf'))
        #     wheel.setVelocity(0.0)
        #     self.wheels.append(wheel)
        self.desired_x_distance = self.parms['desired_x_distance']
        self.desired_y_distance = self.parms['desired_y_distance']
        self.desired_angle = self.parms['desired_angle']
        self.delta_x = self.parms['delta_x']
        self.delta_y = self.parms['delta_y']
        self.delta_angle = self.parms['delta_angle']
        if self.parms['use_done_delta']:
            self.done_delta_distance = self.parms['done_delta_distance']
            self.done_delta_angle = self.parms['done_delta_angle']
        else:
            self.done_delta_distance = 0.0
            self.done_delta_angle = 0.0
        if eval:
            self.max_episode_steps = self.parms['test_each_episode_steps']
        else:
            self.max_episode_steps = self.parms['max_episode_steps']
        self.episode_score = 0
        self.episode_steps = 0
        self.episode_score_list = []
        self.follower = self.getFromDef('follower')
        self.leader = self.getFromDef('leader')
        self.simulationSetMode(2)
        self.console = Console()
        self.follower_low_bound = np.array(self.parms['follower_low_bound'])
        self.follower_high_bound = np.array(self.parms['follower_high_bound'])
        self.leader_low_bound = np.array(self.parms['leader_low_bound'])
        self.leader_high_bound = np.array(self.parms['leader_high_bound'])

        self.lib = ctypes.CDLL('/home/lmf/文档/youbot/libraries/youbot_control/src/libbase.so')
        self.lib.wb_robot_init()
        self.lib.base_init()
        self.lib.set_speed.argtypes = [ctypes.c_double]
        self.lib.set_max_speed.argtypes = [ctypes.c_double]
        self.lib.set_max_omega.argtypes = [ctypes.c_double]
        self.lib.base_move.argtypes = [ctypes.c_double,ctypes.c_double,ctypes.c_double]
        
        self.lib.set_max_speed(self.follower_high_bound[0])
        self.lib.set_max_omega(self.follower_high_bound[-1])

        self.init_leader_translation = self.parms[self.parms['env_name']]['leader']['translation']
        self.init_leader_rotation = self.parms[self.parms['env_name']]['leader']['rotation']

        self.init_follower_translation = self.parms[self.parms['env_name']]['follower']['translation']
        self.init_follower_rotation = self.parms[self.parms['env_name']]['follower']['rotation']
        self.custom_noise = None
        

        if self.observation_type == 'color_image':
            self.color_camera = self.getDevice('color_camera')
            self.img_width = self.parms['observation_shape']['width']
            self.img_height = self.parms['observation_shape']['height']
            self.color_camera.enable(self.timestep)
        elif self.observation_type == 'lidar':
            self.lidar = self.getDevice('Velodyne VLP-16')
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()
        self.info = dict(
            relation_pos=None,
            relative_angle = None,
            image = None,
            follower_vx=None,
            followr_vy=None,
            followr_omega=None,
            leader_vx=None,
            leader_vy=None,
            leader_omega=None,
            status=None
                            )
    def reset(self):
        # super(RobotSupervisorEnv,self).reset()
        self.reset_sensors()
        self.simulationResetPhysics()
        self.lib.base_reset()
        self.status = None
        self.episode_steps = 0
        follower_trans = self.follower.getField('translation')
        # follower_trans.setSFVec3f([-1.9, 1.86, 0.100183])
        follower_trans.setSFVec3f(self.init_follower_translation)
        follower_rotation = self.follower.getField('rotation')
        # follower_rotation.setSFRotation([-0.16617540677112785, -0.9860875011929788, 0.004144173673640147, 0.02756472754243368])
        follower_rotation.setSFRotation(self.init_follower_rotation)

        leader_trans = self.leader.getField('translation')
        # leader_trans.setSFVec3f([5.064146336960932e-06, 1.76, 0.100183])
        leader_trans.setSFVec3f(self.init_leader_translation)
        leader_rotation = self.leader.getField('rotation')
        # leader_rotation.setSFRotation([-0.1811567821711096, 0.37915640200991574, 0.9074263843906591, 0.015331029104798504])
        leader_rotation.setSFRotation(self.init_leader_rotation)



        
        return self.get_default_observation()

    def reset_sensors(self):
        if self.observation_type == 'color_image':
            self.color_camera.enable(self.timestep)
            camera = self.getFromDef('color_camera')
            noise = camera.getField('noise')
            noise_value = min(0.01 + (self.parms['max_camera_noise']-0.01)*self.k/self.parms['total_episodes'],self.parms['max_camera_noise'])
            if self.custom_noise is None:
                noise.setSFFloat(noise_value)
            else:
                noise.setSFFloat(self.custom_noise)

            self.camera_noise = noise.getSFFloat()
        # if self.observation_type == 'lidar':
        #     self.lidar.enable(self.timestep)
        #     self.lidar.enablePointCloud()
        # self.leader.restartController()
    def get_observations(self):
        # obtain image obs
        
        if self.observation_type == 'color_image':
                
            if self.parms['use_random_mask']:
                img = np.frombuffer(self.color_camera.getImage(),dtype=np.uint8).reshape((self.img_height,self.img_width,-1))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = random_occlusion(img)
                img = random_bright(img)
                img = random_contrast(img)
                img = random_gaussian_blur(img)
                # img = img.reshape((-1,self.img_height,self.img_width))
            if self.parms['use_snow']:
                img = np.frombuffer(self.color_camera.getImage(),dtype=np.uint8).reshape((self.img_height,self.img_width,-1))
        
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = add_snow(img,self.parms['snow_severity'])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if self.episode_steps == 1:
                    cv2.imwrite('images/{}_webtos_snow_severity_{}.png'.format(self.parms['env_name'],self.parms['snow_severity']),img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
            else:
                img = np.frombuffer(self.color_camera.getImage(),dtype=np.uint8).reshape((self.img_height,self.img_width,-1))
                if self.episode_steps == 1:
                    cv2.imwrite('images/{}_webtos_color_extrem_noise_{}.png'.format(self.parms['env_name'],self.camera_noise),img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            img = img/255
            img = np.expand_dims(img,axis=0)
            self.info['image'] = img
            # img_s = np.reshape(img,(120,210,3))
        # obtain pose obs        
        relative_pos,relative_angle = obtain_relative_pose(self.follower,self.leader,self.info)
        self.info['relative_pos'] = relative_pos
        self.info['relative_angle'] = relative_angle
        # obstain velocity obs
        velocity = self.follower.getVelocity()

        orientation = self.follower.getOrientation()  # 3x3 rotation matrix
        local_velocity = np.dot(np.linalg.inv(np.array(orientation).reshape(3, 3)), np.array(velocity[:3]))
        follower_vx = local_velocity[0]
        follower_vy = local_velocity[1]
        follower_omega = velocity[-1]

        leader_velocity = self.leader.getVelocity()

        leader_orientation = self.leader.getOrientation()  # 3x3 rotation matrix
        leader_local_velocity = np.dot(np.linalg.inv(np.array(leader_orientation).reshape(3, 3)), np.array(leader_velocity[:3]))
        leader_vx = leader_local_velocity[0]
        leader_vy = leader_local_velocity[1]
        leader_omega = leader_velocity[-1]
        

        follower_vel = self.real_action_to_normal((follower_vx,follower_vy,follower_omega))
        low_features = np.concatenate([self.normal_relative_pos_angle(relative_pos,relative_angle),follower_vel],-1)
        self.info['follower_vx'] = follower_vx
        self.info['follower_vy'] = follower_vy
        self.info['follower_omega'] = follower_omega
        self.info['leader_vx'] = leader_vx
        self.info['leader_vy'] = leader_vy
        self.info['leader_omega'] = leader_omega
        self.info['low_features'] = low_features
        if self.observation_type == 'low_features':
            img = None
        return img,low_features
        
    def get_default_observation(self):
        # return [0.0 for _ in range(self.observation_space.shape[0])]
        return self.get_observations()
    def get_reward(self,action=None):
        if self.info['status'] == 0:
                reward = 100.0
        elif self.info['status'] == 4:
            if self.parms['reward_type'] == 'linear_errors':
                # self.relative_pos,self.relative_angle = obtain_relative_pose(self.follower,self.leader)
                normal_x = abs(self.info['relative_pos'][0]-self.desired_x_distance)/self.delta_x
                normal_y = abs(self.info['relative_pos'][1]-self.desired_y_distance)/self.delta_y
                normal_angle = abs(self.info['relative_angle']-self.desired_angle)/self.delta_angle
                reward = 1 - (normal_x+normal_y+normal_angle)/3
            elif self.parms['reward_type'] == 'constant':
                reward = 1.0
            elif self.parms['reward_type'] == 'ln':
                err_x = abs(self.info['relative_pos'][0]-self.desired_x_distance)
                err_y = abs(self.info['relative_pos'][1]-self.desired_y_distance)
                err_angle = abs(self.info['relative_angle']-self.desired_angle)
                reward1 = 2-np.exp(np.log(2)*err_x/0.3)
                reward2 = 2-np.exp(np.log(2)*err_y/0.3)
                reward3 = 2-np.exp(np.log(2)*err_angle/0.6)
                reward = (reward1 + reward2 + reward3)/3
                # print('errx:',err_x,'reward1:',reward1)
                # print('erry:',err_y,'reward2:',reward2)
                # print('errang:',err_angle,'reward3:',reward3)
            else:
                self.console.print('Reward type not suitable !',style='red')
        else:
            reward = -100.0

        return reward
    def is_done(self):
        if self.episode_steps >= self.max_episode_steps:
            status = 0
            self.console.print('Scuessful done one episode!',style='green')
            self.info['status'] = status
            return True
        elif self.delta_x + self.done_delta_distance < abs(self.info['relative_pos'][0]-self.desired_x_distance):
            status = 1
            self.console.print('Out limit x distance !',style='yellow')
            self.info['status'] = status
            return True
        elif self.delta_y + self.done_delta_distance< abs(self.info['relative_pos'][1]-self.desired_y_distance):
            status = 2
            self.console.print('Out limit y distance !',style='yellow')
            self.info['status'] = status
            return True
        elif self.delta_angle + self.done_delta_angle < abs(self.info['relative_angle']-self.desired_angle):
            status = 3
            self.console.print('Out limit angle !',style='yellow')
            self.info['status'] = status
            return True
        else:
            status = 4
            self.info['status'] = status
        return False
    def solved(self):
        if len(self.episode_score_list) > 250:
            if np.mean(self.episode_score_list[-100:]):
                return True
        return False
    def get_info(self):
        return None
    def render(self,mode='human'):
        pass
    def apply_action(self, action):
        if self.parms['action_space'] == 'continous':
            action = self.normal_action_to_real(action)
            # set_mecanum_wheel_speeds(action[0],action[1],action[2],self.wheels)
            self.lib.base_move(action[0],action[1],action[2])
        elif self.parms['action_space'] == 'discrete':
            if self.parms['action_dim'] == 7:
                if action == 0:
                    pass
                elif action == 1:
                    self.lib.base_forwards_increment()
                elif action == 2:
                    self.lib.base_backwards_increment()
                elif action == 3:
                    self.lib.base_turn_left_increment()
                elif action == 4:
                    self.lib.base_turn_right_increment()
                elif action == 5:
                    self.lib.base_strafe_left_increment()
                elif action == 6:
                    self.lib.base_strafe_right_increment()
                else:
                    self.console.print('the action number max is 6, but now is {}'.format(action),style='red')

            elif self.parms['action_dim'] == 19:
                if action == 0:
                    pass
                elif action == 1:
                    self.lib.base_forwards_increment()
                elif action == 2:
                    self.lib.base_backwards_increment()
                elif action == 3:
                    self.lib.base_turn_left_increment()
                elif action == 4:
                    self.lib.base_turn_right_increment()
                elif action == 5:
                    self.lib.base_strafe_left_increment()
                elif action == 6:
                    self.lib.base_strafe_right_increment()
                elif action == 7:
                    self.lib.base_forwards_strafe_left_increment()
                elif action == 8:                
                    self.lib.base_forwards_strafe_right_increment()
                elif action == 9:                
                    self.lib.base_forwards_turn_left_increment()
                elif action == 10:                
                    self.lib.base_forwards_turn_right_increment()
                elif action == 11:                
                    self.lib.base_forwards_strafe_left_turn_left_increment()
                elif action == 12:                
                    self.lib.base_forwards_strafe_left_turn_rihgt_increment()
                elif action == 13:                
                    self.lib.base_forwards_strafe_right_turn_left_increment()
                elif action == 14:                
                    self.lib.base_forwards_strafe_right_turn_rihgt_increment()
                elif action == 15:                
                    self.lib.base_strafe_left_turn_left_increment()
                elif action == 16:                
                    self.lib.base_strafe_left_turn_rihgt_increment()
                elif action == 17:                
                    self.lib.base_strafe_right_turn_left_increment()
                elif action == 18:                
                    self.lib.base_strafe_right_turn_rihgt_increment()
                else:
                    self.console.print('the action number max is 18, but now is {}'.format(action),style='red')
        self.episode_steps += 1
    def normal_action_to_real(self,normal_action):
        normal_action = np.array(normal_action)
        real_action = self.follower_low_bound + (normal_action - (-1.0)) * (
            (self.follower_high_bound - self.follower_low_bound) / 2.0)
        real_action = np.clip(real_action, self.follower_low_bound, self.follower_high_bound)
        return real_action
    def real_action_to_normal(self,real_action):
        real_action = np.array(real_action)
        if len(real_action) == 6:
            normal_act = -1 + 2*(real_action-np.concatenate([self.follower_low_bound,self.leader_low_bound],-1))/(np.concatenate([self.follower_high_bound,self.leader_high_bound],-1)-np.concatenate([self.follower_low_bound,self.leader_low_bound],-1))
        elif len(real_action) == 3:
            normal_act = -1 + 2*(real_action-self.follower_low_bound)/(self.follower_high_bound-self.follower_low_bound)
        else:
            self.console.print('Real action desired 6 or 3, but received other!!!',style='red')

        return normal_act
    def normal_relative_pos_angle(self,relative_pos,relative_angle):
        normal_relative_x = (relative_pos[0]-self.desired_x_distance)/self.delta_x
        normal_relative_y = (relative_pos[1]-self.desired_x_distance)/self.delta_y #x 修改
        normal_relative_angle = (relative_angle-self.desired_x_distance)/self.delta_angle# 修改
        return (normal_relative_x,normal_relative_y,normal_relative_angle)
    
