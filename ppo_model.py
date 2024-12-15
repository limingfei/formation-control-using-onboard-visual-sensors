import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal
import numpy as np

class Actor(nn.Module):
    def __init__(self, parms):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(parms['state_dim'], 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, parms['action_dim'])
        self.activate_func = [nn.ReLU(), nn.Tanh()][parms['use_tanh']]  # Trick10: use tanh

        if parms['use_orthogonal_init']:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        a_prob = torch.softmax(self.fc3(s), dim=1)
        return a_prob

class Critic(nn.Module):
    def __init__(self, parms):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(parms['state_dim'], 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][parms['use_tanh']]  # Trick10: use tanh

        if parms['use_orthogonal_init']:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class PpoModel(nn.Module):
    def __init__(self, parms):
        super().__init__()
        c,h,w = parms['observation_shape']['channel'],parms['observation_shape']['height'],parms['observation_shape']['width'],
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor_fc1 = layer_init(nn.Linear(256+3, 64))
        self.actor_fc2 = layer_init(nn.Linear(64,64))
        self.actor = layer_init(nn.Linear(64, parms['action_dim']), std=0.01)

        self.critic_fc1 = layer_init(nn.Linear(256, 64))
        self.critic_fc2 = layer_init(nn.Linear(64,64))
        self.critic = layer_init(nn.Linear(64, 3), std=1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][parms['use_tanh']]
    def forward(self, x,low_features, action=None,use_softmax=True):
        hidden = self.network(x)  # "bhwc" -> "bchw"
        s = torch.concat([hidden,low_features],dim=-1)

        s = self.activate_func(self.actor_fc1(s))
        s = self.activate_func(self.actor_fc2(s))
        if use_softmax:
            logits = torch.softmax(self.actor(s),dim=1)
        else:
            logits = self.actor(s)
        # probs = Categorical(logits=logits)
        # if action is None:
        #     action = probs.sample()
        return logits
    
    def get_res(self, x):
        hidden = self.network(x) # "bhwc" -> "bchw"        

        s = self.activate_func(self.critic_fc1(hidden))
        s = self.activate_func(self.critic_fc2(s))
        s = self.critic(s)

        return s # "bhwc" -> "bchw"

    
class Visual_model(nn.Module):
    def __init__(self):
        super().__init__()
        c,h,w = 1,120,210
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
    def forward(self, x):
        hidden = self.network(x)  
        return hidden
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
model = Visual_model().to(device)
from torchsummary import summary
summary(model,(1,120,210))

    
