import os
import glob
from DotDict import DotDict
from torch.utils.tensorboard import SummaryWriter

class Manager:

    def __init__(self, config, parent_path):
        self.config = config
        self.parent_path = parent_path
        self.experiment_path = self.return_and_make_available_path(self.parent_path) # make variable
        self.summary_writer = self.return_summary_writer(self.experiment_path) # get summary writer
        
        self.record_experiment_info(self.config, self.experiment_path) # saves experiment info

        self.episode_total_reward = 0
        self.node_selected_num = [0] * self.config.node_num
        self.void_selected_num = 0
        

    def record_experiment_info(self, config, path):
        info = ""
        for key in config:
            info += f"{key} : {config[key]}\n"

        with open(f'{path}/info.txt', 'w') as f:
            f.write(f'{info}')

    def return_and_make_available_path(self, parent_path):
        os.chdir(parent_path)
        folder_list = glob.glob("history*")
        experiment_path = "history" + str(len(folder_list))
        os.mkdir(experiment_path)
        return experiment_path

    def return_summary_writer(self, experiment_path):
        writer = SummaryWriter(experiment_path)

        return writer

    def record_summary(self, name, scalar, episode = None):
        if episode == None:
            self.summary_writer.add_scalar(name + "/train", scalar)
        else:
            self.summary_writer.add_scalar(name + "/train", scalar, episode)

    def update_episode_total_reward(self, reward):
        self.episode_total_reward += reward

    def get_episode_total_reward(self):
        return self.episode_total_reward
    
    def record_episode_total_reward(self, episode):
        self.record_summary("episode_total_reward/train", self.episode_total_reward, episode)
        self.episode_total_reward = 0

    def update_node_selected_num(self, node_number):
        self.node_selected_num[node_number] += 1
    
    def get_node_selected_num(self):
        return self.node_selected_num
    
    def record_node_selected_num(self, episode):
        for i in range(self.config.node_num):
            node_tag = "node/" + str(i) + "/train"
            self.record_summary(node_tag, self.node_selected_num[i], episode)

        self.node_selected_num = [0] * self.config.node_num
    
    def update_void_selected_num(self):
        self.void_selected_num += 1
    
    def get_void_selected_num(self):
        return self.void_selected_num
    
    def record_void_selected_num(self, episode):
        self.record_summary("node/void/train", self.void_selected_num, episode)

        self.void_selected_num = 0