from ppo_model import PpoModel
import cv2
from webots_env import WebotsEnv
from ppo_agent import PpoAgent
import torch
from utils import Normalization,AttentionVis
from tensorboardX import SummaryWriter
import numpy as np
import os
import yaml

np.random.seed(10)
param_path = 'env_config_3.yaml'

with open('./params/best/best_mean_std.yaml','r') as file:
    mean_std = yaml.safe_load(file)
with open('./params/final/final_mean_std.yaml', 'r', encoding='utf-8') as file:
    mean_std_low = yaml.load(file, Loader=yaml.UnsafeLoader)
mean_array = np.array(mean_std_low['mean'])
std_array = np.array(mean_std_low['std'])

env = WebotsEnv(parms_path=param_path,eval=True)
actor_model = PpoModel(env.parms)

state_norm = Normalization(shape=(env.parms['observation_shape']['channel'],env.parms['observation_shape']['height'],env.parms['observation_shape']['width']))
state_norm.running_ms.mean = np.array(mean_std['mean'])
state_norm.running_ms.std = np.array(mean_std['std'])
state_norm_low = Normalization(shape=6)
state_norm_low.running_ms.mean = mean_array
state_norm_low.running_ms.std = std_array
load_model_path = './params/best/best.pth'


model_dict = torch.load(load_model_path)

parms = env.parms

actor_model.load_state_dict(model_dict)
# att_vis = AttentionVis(actor_model)
parms['load_model_path'] = load_model_path
env.simulationSetMode(1)
env.k = 20
env.custom_noise = 0.5
img,low_features = env.reset()
total_steps = 0
time_path = load_model_path.split('/')[-2]
log_path = os.path.join('test_logs','extrem_noise',env.parms['action_space'],'mask_train_greedy','{}'.format(time_path),'env_{}_noise_{}_dy'.format(env.parms['env_name'],round(env.camera_noise,3)))
writer = SummaryWriter(log_path)
with open(os.path.join(log_path,'config.yaml'), 'w') as file:
    yaml.dump(env.parms, file, default_flow_style=False, allow_unicode=True)
# writer = SummaryWriter(os.path.join('test_logs',env.parms['action_space'],'noise_{}'.format(round(env.camera_noise,3))))
# writer = SummaryWriter(os.path.join('test_logs',env.parms['action_space'],'no_mask','noise_{}'.format(round(env.camera_noise,3))))
total_distance = 0.0
while total_steps <= env.parms['test_each_episode_steps']:
    done = False
    total_steps += 1
    # img = state_norm(img,update=False)
    low_features = state_norm_low(low_features,update=False)
    img_raw = img.copy()
    img = torch.FloatTensor(img)
    low_features = torch.FloatTensor(low_features[3:])
    img = torch.reshape(img,shape=(1,env.parms['observation_shape']['channel'],env.parms['observation_shape']['height'],env.parms['observation_shape']['width']))
    low_features = torch.reshape(low_features,shape=(1,-1))
    action_prob = actor_model(img,low_features).detach().cpu().numpy().flatten()
    action = np.argmax(action_prob)
    (img,low_features),reward,done,info = env.step(action)
    total_distance += np.linalg.norm(np.array(env.info['relative_pos'][:2]))-1.5
    writer.add_scalar('distance_x',env.info['relative_pos'][0],total_steps)
    writer.add_scalar('distance_y',env.info['relative_pos'][1],total_steps)
    writer.add_scalar('angles',env.info['relative_angle'],total_steps)
    writer.add_scalar('follower_vx',env.info['follower_vx'],total_steps)
    writer.add_scalar('follower_vy',env.info['follower_vy'],total_steps)
    writer.add_scalar('follower_omega',env.info['follower_omega'],total_steps)
    writer.add_scalar('leader_vx',env.info['leader_vx'],total_steps)
    writer.add_scalar('leader_vy',env.info['leader_vy'],total_steps)
    writer.add_scalar('leader_omega',env.info['leader_omega'],total_steps)
    writer.add_scalar('leader_x',env.info['leader_x'],total_steps)
    writer.add_scalar('leader_y',env.info['leader_y'],total_steps)
    writer.add_scalar('follower_x',env.info['follower_x'],total_steps)
    writer.add_scalar('follower_y',env.info['follower_y'],total_steps)


