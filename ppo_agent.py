import parl
import torch
import yaml
import os
from torch.utils.data import DataLoader
from utils import MyDataset
class PpoAgent(parl.Agent):
    def __init__(self, algorithm,parms):
        super(PpoAgent,self).__init__(algorithm)
        self.device = torch.device(parms['device'])
        # self.device = 'cpu'
        self.parms = parms
    def predict(self,obs,low_features):
        if self.parms['observation_type'] == 'color_image':
            obs = torch.FloatTensor(obs.reshape(1,self.parms['observation_shape']['channel'],self.parms['observation_shape']['height'],self.parms['observation_shape']['width'])).to(self.device)
            low_features = torch.FloatTensor(low_features.reshape(1,3)).to(self.device)
            action = self.alg.predict(obs,low_features)
        elif self.parms['observation_type'] == 'low_features':
            obs = torch.FloatTensor(obs.reshape(1,6)).to(self.device)
            action = self.alg.predict(obs,low_features)
        return action
    def sample(self,obs,low_features):
        if self.parms['observation_type'] == 'color_image':
            obs = torch.FloatTensor(obs.reshape(1,self.parms['observation_shape']['channel'],self.parms['observation_shape']['height'],self.parms['observation_shape']['width'])).to(self.device)
            low_features = torch.FloatTensor(low_features.reshape(1,3)).to(self.device)
            action = self.alg.sample(obs,low_features)
        elif self.parms['observation_type'] == 'low_features':
            obs = torch.FloatTensor(obs.reshape(1,6)).to(self.device)
            action = self.alg.sample(obs,low_features)
        return action
    def learn(self,all_img,all_low_features,all_teacher_action,all_relative_pose):
        dataset = MyDataset(all_img,all_low_features,all_teacher_action,all_relative_pose)
        dataloader = DataLoader(dataset, batch_size=self.parms['mini_batch_size'], shuffle=True, num_workers=4)
        for batch_idx, (img,low_features, targets,relative_pose) in enumerate(dataloader):
            img = torch.FloatTensor(img).to(self.device)
            low_features = torch.FloatTensor(low_features).to(self.device)
            relative_pose = torch.FloatTensor(relative_pose).to(self.device)
            targets = torch.LongTensor(targets).to(self.device)
            loss = self.alg.learn(img,low_features,targets,relative_pose)
        return loss,batch_idx+1
    def save(self, save_path_actor):
        actor_model = self.alg.actor_model
        sep = os.sep
        actor_dirname = sep.join(save_path_actor.split(sep)[:-1])
        if actor_dirname != '' and not os.path.exists(actor_dirname):
            os.makedirs(actor_dirname)
        torch.save(actor_model.state_dict(), save_path_actor)

    def restore(self, save_path_actor, map_location=None):
        
        actor_model = self.alg.model
        actor_checkpoint = torch.load(save_path_actor, map_location=map_location)
        actor_model.load_state_dict(actor_checkpoint)