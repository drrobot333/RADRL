from DotDict import DotDict
from Communicator import Communicator
import json
import torch
import copy
from torch_geometric.data import Data
import itertools

PIPE_NAME_BASE = "\\\\.\\pipe\\worker_right_latency_sub_trace"

class Simulator:
    def __init__(self, args, buffer_size):
        self.time_step = 1
        self.episode = 1

        self.node_num = args.node_num
        self.buffer_size = buffer_size

        # for trial in itertools.count(start=1): # 무한반복함

        #     self.pipe_name = f"{PIPE_NAME_BASE}_{self.node_num}_{trial}"
            
        #     self.communicator = Communicator(self.pipe_name, self.buffer_size)

        #     if self.communicator.is_communicator_connected():
        #         print("OMNeT++ connected")
        #         print(f"Pipe_name : {self.pipe_name}")
        #         break

        self.pipe_name = f"{PIPE_NAME_BASE}_{self.node_num}_{args.process_num}"
        self.communicator = Communicator(self.pipe_name, self.buffer_size)
        if self.communicator.is_communicator_connected():
            print("OMNeT++ connected")
            print(f"Pipe_name : {self.pipe_name}")
        else:
            raise "pipe error: Pipe is already open"
        
        self.network_info = DotDict({})
        self.pre_state = DotDict({})

    def open_omnet(self):
        pass


    def get_initial_info(self):
        initial_message = self.communicator.get_omnet_message()
        network_info_json = json.loads(initial_message)

        self.network_info.model_num = int(network_info_json['modelNum'])
        self.network_info.available_job_num = int(network_info_json['availableJobNum'])
        self.network_info.node_num = int(network_info_json['nodeNum'])
        self.network_info.job_waiting_length = int(network_info_json['jobWaitingQueueLength'])
        self.network_info.adjacency = torch.tensor(eval(network_info_json['adjacencyList']), dtype=torch.long)
        self.network_info.episode_length = int(network_info_json['episode_length'])
        self.network_info.node_capacity = network_info_json['node_capacity']
        self.network_info.job_generate_rate = network_info_json['job_generate_rate']

        return self.network_info
    
    def step(self, action):
        if action == "void": # void action
            self.communicator.send_omnet_message("void")

        else: # none void action, offloading
            self.communicator.send_omnet_message(action)
                    
            #print("action finish.")
            if(self.get_response() == "ok"):
                pass

    def start_simulator(self):
        self.communicator.send_omnet_message("init")
    
    def get_response(self):
        return self.communicator.get_omnet_message()
    
    def get_episode_result(self):
        return self.communicator.get_omnet_message()
    
    def get_time_step(self):
        return self.time_step

    def set_time_step(self, time_step):
        self.time_step = time_step

    def get_episode(self):
        return self.episode

    def set_episode(self, episode):
        self.episode = episode

    def send_response(self):
        self.communicator.send_omnet_message("ok")

    def return_env_status(self):
        return self.communicator.get_omnet_message()
    
    def return_state(self):
        self.send_response() # state를 받기 위한 답장

        pre_network = self.communicator.get_omnet_message()
        cur_network = self.communicator.get_omnet_message()

        self.send_response()

        if len(self.pre_state) != 0: # 동일한 time step 내에서 scheduling 하는 것 이므로 pre_state 반환 -> action을 선택하는 과정에서 정확하게 pre_state에 기록해야 함, episode가 끝나면 pre_state 초기화 해야함
            return copy.deepcopy(self.pre_state)
        
        if len(pre_network) == 0:
            pre_network = copy.deepcopy(cur_network)

        pre_network = json.loads(pre_network) # state 받았으므로 action 하면됨.
        cur_network = json.loads(cur_network) # state 받았으므로 action 하면됨.

        if len(pre_network) == 0: # timestep = 0
            pre_network = cur_network

        cur_state = {"pre_network":{}, "cur_network":{}}

        cur_state["pre_network"]["node_waiting_state"]       = torch.tensor(eval(str(pre_network['nodeState'])), dtype=torch.float)
        cur_state["pre_network"]["node_processing_state"]    = torch.tensor(eval(pre_network['nodeProcessing']), dtype=torch.float)
        cur_state["pre_network"]["link_state"]               = torch.tensor(eval(pre_network['linkWaiting']), dtype=torch.float)
        cur_state["pre_network"]["job_waiting_state"]        = torch.tensor(eval(pre_network['jobWaiting']), dtype=torch.float).view(1, -1)
        cur_state["pre_network"]["activated_job_list"]       = eval(pre_network['activatedJobList'])
        cur_state["pre_network"]["is_action"]                = int(pre_network['isAction'])
        cur_state["pre_network"]["reward"]                   = float(pre_network['reward'])
        cur_state["pre_network"]["average_latency"]          = float(pre_network['averageLatency'])
        cur_state["pre_network"]["complete_job_num"]         = int(pre_network['completeJobNum'])
        cur_state["pre_network"]["sojourn_time"]             = float(pre_network['sojournTime'])
        #cur_state["pre_network"]["start_latency"]            = float(pre_network["startLatency"])
        cur_state["pre_network"]["network_state"]            = Data(x=torch.concat([cur_state["pre_network"]["node_waiting_state"], cur_state["pre_network"]["node_processing_state"]], dim=1), edge_attr=cur_state["pre_network"]["link_state"], edge_index=self.network_info.adjacency)
        cur_state["pre_network"]["job_index"]                = int(pre_network['jobIndex'])

        cur_state["cur_network"]["node_waiting_state"]       = torch.tensor(eval(str(cur_network['nodeState'])), dtype=torch.float) # 
        cur_state["cur_network"]["node_processing_state"]    = torch.tensor(eval(cur_network['nodeProcessing']), dtype=torch.float) #
        cur_state["cur_network"]["link_state"]               = torch.tensor(eval(cur_network['linkWaiting']), dtype=torch.float)
        cur_state["cur_network"]["job_waiting_state"]        = torch.tensor(eval(cur_network['jobWaiting']), dtype=torch.float).view(1, -1)
        cur_state["cur_network"]["activated_job_list"]       = eval(cur_network['activatedJobList'])
        cur_state["cur_network"]["is_action"]                = int(cur_network['isAction'])
        cur_state["cur_network"]["reward"]                   = float(cur_network['reward'])
        cur_state["cur_network"]["average_latency"]          = float(cur_network['averageLatency'])
        cur_state["cur_network"]["complete_job_num"]         = int(cur_network['completeJobNum'])
        cur_state["cur_network"]["sojourn_time"]             = float(cur_network['sojournTime'])
        #cur_state["cur_network"]["start_latency"]            = float(cur_network["startLatency"])
        cur_state["cur_network"]["network_state"]            = Data(x=torch.concat([cur_state["cur_network"]["node_waiting_state"], cur_state["cur_network"]["node_processing_state"]], dim=1), edge_attr=cur_state["cur_network"]["link_state"], edge_index=self.network_info.adjacency)
        cur_state["cur_network"]["job_index"]                = int(cur_network['jobIndex'])

        return copy.deepcopy(cur_state)
    
    def return_estimated_state(self, cur_state, action, subtask_info, subtask_index): # pre_state를 저장하고, 새로운 state를 만드는 것이 목표
        self.pre_state = copy.deepcopy(cur_state)
        next_state = copy.deepcopy(cur_state) # 새로운 cur_state를 만들기 위해 이전 state를 복사하고, 일부만 변경하기 위함임

        # next_state["cur_network"]["node_waiting_state"][action][3] += subtask_info[subtask_index] # node의 subtask위치에 job 계산량을 추가
        next_state["cur_network"]["node_waiting_state"][action][cur_state["cur_network"]["job_index"] * self.network_info.model_num + subtask_index] += subtask_info[subtask_index]

        next_state["pre_network"]["network_state"] = Data(x=torch.concat([cur_state["pre_network"]["node_waiting_state"], cur_state["pre_network"]["node_processing_state"]], dim=1), edge_attr=next_state["pre_network"]["link_state"], edge_index=self.network_info.adjacency) # adjacency를 torch로 변경해야 함
        next_state["cur_network"]["network_state"] = Data(x=torch.concat([cur_state["cur_network"]["node_waiting_state"], cur_state["cur_network"]["node_processing_state"]], dim=1), edge_attr=next_state["cur_network"]["link_state"], edge_index=self.network_info.adjacency) # adjacency를 torch로 변경해야 함

        return next_state