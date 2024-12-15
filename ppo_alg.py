import torch
import numpy as np
import torch.nn.functional as F
import parl
from torch.distributions import Categorical
from parl.utils.utils import check_model_method
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
class PpoAlg(parl.Algorithm):
    def __init__(self,
                 actor_model,
                 max_action=None,
                 batch_size=None,
                 mini_batch_size=None,
                 max_train_steps=None,
                 lr_actor=None,
                 lr_critic=None,
                 gamma=None,
                 lamda=None,
                 epsilon=None,
                 k_epochs=None,
                 entorpy_coef=None,
                 set_adam_eps=None,
                 use_grad_clip=None,
                 use_lr_decay=None,
                 use_adv_norm=None,
                 device_id=None
                 ):
        
        # check_model_method(actor_model, 'get_dist', self.__class__.__name__)
        assert isinstance(max_action,float)
        assert isinstance(batch_size,int)
        assert isinstance(mini_batch_size,int)
        assert isinstance(max_train_steps,int)
        assert isinstance(lr_actor,float)
        assert isinstance(lr_critic,float)
        assert isinstance(gamma,float)
        assert isinstance(lamda,float)
        assert isinstance(epsilon,float)
        assert isinstance(k_epochs,int)
        assert isinstance(entorpy_coef,float)
        assert isinstance(set_adam_eps,bool)
        assert isinstance(use_grad_clip,bool)
        assert isinstance(use_lr_decay,bool)
        assert isinstance(use_adv_norm,bool)
        assert isinstance(device_id,str)
        
        self.max_train_steps = max_train_steps
        self.max_action = max_action
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.lamda = lamda
        self.epsilon = epsilon
        self.k_epochs = k_epochs
        self.entropy_coef = entorpy_coef
        self.set_adam_eps = set_adam_eps
        self.use_grad_clip = use_grad_clip
        self.use_lr_decay = use_lr_decay
        self.use_adv_norm = use_adv_norm

        self.device = torch.device(device_id)

        self.actor_model = actor_model.to(self.device)
        if self.set_adam_eps:
            self.optimizer = torch.optim.Adam(self.actor_model.parameters(),lr=self.lr_actor,eps=1e-5)
        else:
            self.optimizer = torch.optim.Adam(self.actor_model.parameters(),lr=self.lr_actor)
    def predict(self,obs,low_features):
        action_prob = self.actor_model(obs,low_features).detach().cpu().numpy().flatten()
        action = np.argmax(action_prob)
        return action
    def sample(self,obs,low_features):
        with torch.no_grad():
            dist = Categorical(probs=self.actor_model(obs,low_features))
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        return action.cpu().numpy()[0]
    
    def learn(self,imgs,low_features,teacher_action,relative_pose):
        predict = self.actor_model(imgs,low_features,use_softmax=False)
        policy_loss = F.cross_entropy(predict,teacher_action).mean()
        low_p = self.actor_model.get_res(imgs)
        res_loss = F.mse_loss(low_p,relative_pose).mean()
        loss = 0.7*policy_loss+0.3*res_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(),0.5)
        self.optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(self.total_train_steps)
        return loss
    def lr_decay(self,total_steps):
        lr_now = self.lr_actor*(1-total_steps/self.max_train_steps)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now
                
        

            



