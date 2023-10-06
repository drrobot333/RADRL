import time
from time import strftime, localtime
import json
from collections import deque
import random
import argparse

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from DotDict import DotDict

from Manager import Manager
from Simulator import Simulator
from Agent import PPO
from run import run

argparser = argparse.ArgumentParser()

argparser.add_argument('--node-num', type=int, help='network node number', default=5)
argparser.add_argument('--process-num', type=int, help='process num with same node number', default=1)



BUFFER_SIZE = 200000
PARENT_PATH = "C:/Users/Suhwan/Desktop/suhwan/connection_test/python_agent/experiment/subtask_reward"
args = argparser.parse_args()



env = Simulator(args, BUFFER_SIZE)

network_info = env.get_initial_info()
env.start_simulator() # 입력 끝나면 omnet에 전송


                
config = DotDict({
    "learning_rate"         : 0.00005,
    "gamma"                 : 0.9,
    "entropy_weight"        : 0.0001,
    "entropy_gamma"         : 0.9998,
    "lambda"                : 0.99,
    "eps_clip"              : 0.08,
    "batch_size"            : 256,
    "loss_coef"             : 0.5,
    "job_generate_rate"     : 0.003,
    "is_train"              : True,
    "replay_buffer_size"    : 100,
    "history_learning_time" : 2,
    "current_learning_time" : 2,
    "node_feature_num"      : 2 * (network_info.model_num * network_info.available_job_num),
    "queue_feature_num"     : (network_info.node_num + network_info.model_num) * network_info.job_waiting_length,
    "hidden_feature_num"    : 128,
    "reward_weight"         : 1.0/1,
    "node_num"              : network_info.node_num,
    "model_num"             : network_info.model_num,
    "adjacency"             : network_info.adjacency,
    "lstm_hidden_num"       : 128,
    "cpu_load_balance_time" : 0.1,
    "network_info"          : network_info,
    "path_name"             : PARENT_PATH,
    "T_horizon"             : 1000,
    "link_num"              : 32,
    "state_weight"          : 1.0,
    "our"                   : True,
    "imitation_probability" : 0.0,
    "imitation_gamma"       : 1.0,
    "buffer_size"           : BUFFER_SIZE,
    "load_model"            : False, 
    "model_path"            : "C:/Users/Suhwan/Desktop/suhwan/connection_test/python_agent/experiment/subtask_reward/history2/model_2500.pth",
})

manager = Manager(config, PARENT_PATH)

if __name__ == "__main__":
    run(config, env, manager)