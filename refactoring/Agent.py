import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, global_mean_pool, GraphUNet
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
from collections import deque

class PPO(nn.Module):
    def __init__(self, config):
        super(PPO, self).__init__()
        self.data = [] # 에피소드의 데이터
        
        self.config = config
        self.history = deque(maxlen=self.config["replay_buffer_size"]) # 과거의 데이터

        self.x_mean = 0
        self.x2_mean = 0
        self.sd = 0
        self.reward_sample_num = 0

        self.step = 0
        #self.entropy_normalize_weight = 1/torch.log(torch.tensor(nodeNum))
        self.entropy_normalize_weight = 1

        
        # print(f"self.adjacency : ", self.adjacency.shape)

        self.pi_mlp1 = nn.Sequential(
            nn.Linear(1, self.config["hidden_feature_num"]), 
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"], self.config["hidden_feature_num"]), 
            nn.ReLU(),
            nn.Linear(int(self.config["hidden_feature_num"]), self.config["node_feature_num"] * int(self.config["node_feature_num"])), 
            nn.ReLU()
            )

        self.pi_s_ecc1 = NNConv(self.config["node_feature_num"], int(self.config["node_feature_num"]), self.pi_mlp1, aggr='add')

        self.pi_mlp2 = nn.Sequential(
            nn.Linear(1, self.config["hidden_feature_num"]), 
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"], self.config["hidden_feature_num"]), 
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"],  int(self.config["node_feature_num"]) * int(self.config["node_feature_num"])), 
            nn.ReLU(),
            )
        self.pi_s_ecc2 = NNConv(int(self.config["node_feature_num"]), int(self.config["node_feature_num"]), self.pi_mlp2, aggr='add')

        self.pi_graph_u_net1 = GraphUNet(int(self.config["node_feature_num"]), 32, int(self.config["node_feature_num"]), 2, 0.6)
        self.pi_graph_u_net2 = GraphUNet(int(self.config["node_feature_num"]), 64, int(self.config["node_feature_num"]), 4, 0.8)

        self.pi_backbone = nn.Sequential(
            nn.Linear((int(self.config["node_feature_num"]) * self.config["node_num"] + self.config["queue_feature_num"]) * 2, self.config["hidden_feature_num"]),
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"], self.config["hidden_feature_num"] * 4), 
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"] * 4, self.config["hidden_feature_num"] * 4), 
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"] * 4, self.config["hidden_feature_num"] * 4), 
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"] * 4, self.config["hidden_feature_num"]), 
            nn.ReLU(),
        )

        """
        self.pi_backbone = nn.Sequential(
            nn.Linear((int(node_feature_num//2) + queue_feature_num) * 2, hidden_feature_num),
            nn.ReLU(),
            nn.Linear(hidden_feature_num, hidden_feature_num), # 38부터 적용되는 코드. 38이전은 두 줄 지우고 실행하면됨.
            nn.ReLU(),
            
        )
        """

        self.policy = nn.LSTM(input_size = self.config["hidden_feature_num"], hidden_size = self.config["lstm_hidden_num"], batch_first=True)

        # prob_fc : 각 액션에 대한 확률.
        # self.pi_prob_fc = nn.Linear(self.config["lstm_hidden_num"], self.config["node_num"] + 1)
        
        
        self.pi_prob_fc = nn.Sequential( 
            nn.Linear(self.config["lstm_hidden_num"], self.config["hidden_feature_num"]), # nodeNum + voidAction
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"], self.config["hidden_feature_num"] * 4), 
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"] * 4, self.config["hidden_feature_num"] * 4), 
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"] * 4, self.config["hidden_feature_num"]), 
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"], self.config["node_num"] + 1), # 38부터 적용되는 코드. 38이전은 두 줄 지우고 실행하면됨.
        )


        # prob_fc : 각 액션에 대한 확률.
        self.v_value_fc = nn.Sequential( 
            nn.Linear(self.config["hidden_feature_num"], self.config["hidden_feature_num"]), # nodeNum + voidAction
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"], self.config["hidden_feature_num"] * 4), 
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"] * 4, self.config["hidden_feature_num"] * 4), 
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"] * 4, self.config["hidden_feature_num"]), 
            nn.ReLU(),
            nn.Linear(self.config["hidden_feature_num"], 1), # 38부터 적용되는 코드. 38이전은 두 줄 지우고 실행하면됨.
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.config["learning_rate"])
              
        
    # policy DNN
    def gnn(self, state):
        data, job_waiting_feature = state
        node_feature_1, link_feature_1, adjacency_1 = data[0].x, data[0].edge_attr, data[0].edge_index
        node_feature_2, link_feature_2, adjacency_2 = data[1].x, data[1].edge_attr, data[1].edge_index

        job_waiting_feature_1 = job_waiting_feature[0]
        job_waiting_feature_2 = job_waiting_feature[1]

        if job_waiting_feature_2.device == 'cuda':
            link_feature_1 = link_feature_1.cuda()
            adjacency_1 = adjacency_1.cuda()

            link_feature_2 = link_feature_2.cuda()
            adjacency_2 = adjacency_2.cuda()

        #node_feature_1 = F.normalize(node_feature_1, dim = 0)
        #node_feature_2 = F.normalize(node_feature_2, dim = 0)

        #print(node_feature_1)

        
        # =========================================================================
        node_feature_1 = F.relu(self.pi_s_ecc1(node_feature_1, adjacency_1, link_feature_1))
        node_feature_1 = F.relu(self.pi_graph_u_net1(node_feature_1, adjacency_1))

        node_feature_1 = F.relu(self.pi_s_ecc2(node_feature_1, adjacency_1, link_feature_1))
        node_feature_1 = F.relu(self.pi_graph_u_net2(node_feature_1, adjacency_1))

        data_num_1 = len(node_feature_1) // self.config["node_num"]
        
        #readout_1 = global_mean_pool(node_feature_1, data[0].batch)

        readout_1 = node_feature_1.view(data_num_1, -1)


        concat_1 = torch.cat([readout_1, job_waiting_feature_1], dim=1) # 여기에 job waiting 벡터 붙이기.

        # =========================================================================
        node_feature_2 = F.relu(self.pi_s_ecc1(node_feature_2, adjacency_2, link_feature_2))
        node_feature_2 = F.relu(self.pi_graph_u_net1(node_feature_2, adjacency_2))

        node_feature_2 = F.relu(self.pi_s_ecc2(node_feature_2, adjacency_2, link_feature_2))
        node_feature_2 = F.relu(self.pi_graph_u_net2(node_feature_2, adjacency_2))

        data_num_2 = len(node_feature_2) // self.config["node_num"]

        #vreadout_2 = global_mean_pool(node_feature_2, data[0].batch)

        readout_2 = node_feature_2.view(data_num_2, -1)

        concat_2 = torch.cat([readout_2, job_waiting_feature_2], dim=1) # 여기에 job waiting 벡터 붙이기.

        # =========================================================================

        concat = torch.cat([concat_1, concat_2], dim=1)
        # concat = concat_2 - concat_1

        feature_extract = self.pi_backbone(concat)

        return feature_extract


    def pi(self, state):

        inp, h_c = state

        output, hidden = self.policy(inp, h_c)

        output = self.pi_prob_fc(output)


        # print("output", output.shape)

        prob = F.softmax(output, dim=2)


        # print("prob", prob.shape)
        
        # 아래는 엔트로피 구하는 과정
        log_prob = F.log_softmax(output, dim=2)

        # print("log_prob", log_prob.shape)

        en = - (log_prob * prob)
        entropy = torch.sum(en, dim=2)

        # print("en", en.shape)



        # print(entropy.shape)
        
        return prob, entropy, output, hidden

    # advantage network
    def v(self, state):

        value = self.v_value_fc(state)

        return value
    
    def return_parameter_number(self):
        return sum(p.numel() for p in self.parameters())
    
    def print_parameter_number(self):
        total_params = self.return_parameter_number()
        print(f"Total_params : {total_params}")
        
    def put_data(self, transition):
        self.data.append(transition)

    def clear_data(self):
        self.data = []

    def set_reward(self, reward):
        self.data[-1][3] = reward

    def set_reward(self, reward):
        self.data[-1][3] = reward
        
    def return_link_dict(sel, ad):
        result = {}
        for i in range(len(ad[0])//2):
            result[f'{ad[0][2*i]}{ad[0][2 *(i)+1]}'] = i
            
        return result

    def return_new_mean(self, mean, num, new_data):
        result = (mean * num + new_data) / (num + 1)
        return result
        
    def return_new_sd(self, square_mean, mean):
        result = (square_mean - mean**2)**0.5
        
        return result
        
    def return_normalize_reward(self, reward):
        self.x_mean = self.return_new_mean(self.x_mean, self.reward_sample_num, reward)
        self.x2_mean = self.return_new_mean(self.x2_mean, self.reward_sample_num, reward**2)
        self.sd = self.return_new_sd(self.x2_mean, self.x_mean)
        
        if self.sd == 0:
                z_normalized = 0
        else:  
            z_normalized = (reward - self.x_mean) / self.sd

        self.reward_sample_num += 1
        
        return z_normalized
        
    
    # make_batch, train_net은 맨 위에 코드 기반 링크와 거의 동일합니다.

    def make_batch_history(self):
        sample_index = torch.randperm(len(self.history))[:self.config["batch_size"]]

        if len(sample_index) == 0:
            return -1
        
        sample_index = sample_index[0]

        sampled_data = []
        for sample in self.history[sample_index]:
            sampled_data.append(sample)


        b_size = min(len(sampled_data), self.config["batch_size"])


        network_1_lst = [None] * b_size
        network_2_lst = [None] * b_size
        next_network_1_lst = [None] * b_size
        next_network_2_lst = [None] * b_size
        job_waiting_1_lst = [None] * b_size
        job_waiting_2_lst = [None] * b_size
        next_job_waiting_1_lst = [None] * b_size
        next_job_waiting_2_lst = [None] * b_size
        a_lst = [None] * b_size
        r_lst = [None] * b_size
        prob_a_lst = [None] * b_size
        sojourn_time_lst = [None] * b_size
        action_mask_lst = [None] * b_size
        subtask_index_lst = [None] * b_size

        for idx, transition in enumerate(sampled_data):
            #self.history.append(transition)
            network, job_waiting, a, r, nxt_network, nxt_job_waiting, prob_a, sojourn_time, action_mask, subtask_index = transition

            r_lst[idx] = [r*self.config["reward_weight"]]

            network_1_lst[idx] = network[0]
            network_2_lst[idx] = network[1]

            job_waiting_1_lst[idx] = job_waiting[0].tolist()
            job_waiting_2_lst[idx] = job_waiting[1].tolist()

            a_lst[idx] = [a]

            next_network_1_lst[idx] = nxt_network[0]
            next_network_2_lst[idx] = nxt_network[1]
            
            next_job_waiting_1_lst[idx] = nxt_job_waiting[0].tolist()
            next_job_waiting_2_lst[idx] = nxt_job_waiting[1].tolist()

            prob_a_lst[idx] = [prob_a]
            sojourn_time_lst[idx] = [sojourn_time * 5]

            action_mask_lst[idx] = action_mask
            subtask_index_lst[idx] = [subtask_index]

            if idx == self.config["batch_size"] - 1:
                break
        

        # gnn sample을 배치단위로 inference하려면 이렇게 묶어줘야 함.
        network_1_loader = DataLoader(network_1_lst, batch_size=len(network_1_lst))
        next_network_1_loader = DataLoader(next_network_1_lst, batch_size=len(network_1_lst))
        network_1_batch = next(iter(network_1_loader))
        next_network_1_batch = next(iter(next_network_1_loader))

        network_2_loader = DataLoader(network_2_lst, batch_size=len(network_2_lst))
        next_network_2_loader = DataLoader(next_network_2_lst, batch_size=len(network_2_lst))
        network_2_batch = next(iter(network_2_loader))
        next_network_2_batch = next(iter(next_network_2_loader))

        # print(job_waiting_lst)
        
        # job_waiting = torch.tensor(np.array(job_waiting_lst), dtype=torch.float)
        job_waiting_1 = torch.squeeze(torch.tensor(np.array(job_waiting_1_lst), dtype=torch.float), dim=1)
        job_waiting_2 = torch.squeeze(torch.tensor(np.array(job_waiting_2_lst), dtype=torch.float), dim=1)

        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst, dtype=torch.float)
        sojourn_time = torch.Tensor(sojourn_time_lst)

        next_job_waiting_1 = torch.squeeze(torch.tensor(np.array(next_job_waiting_1_lst), dtype=torch.float), dim=1)
        next_job_waiting_2 = torch.squeeze(torch.tensor(np.array(next_job_waiting_2_lst), dtype=torch.float), dim=1)
        
        prob_a = torch.tensor(prob_a_lst, dtype=torch.float)
        action_mask_lst = torch.tensor(action_mask_lst)

        subtask_index_lst = torch.tensor(subtask_index_lst, dtype=torch.long)
        
        # self.data = []
        return [network_1_batch, network_2_batch], [job_waiting_1, job_waiting_2], a, r, [next_network_1_batch, next_network_2_batch], [next_job_waiting_1, next_job_waiting_2], prob_a, sojourn_time, action_mask_lst, subtask_index_lst, min(self.config["batch_size"], idx + 1)

        
    def make_batch(self):

        b_size = min(len(self.data), self.config["batch_size"])

        network_1_lst = [None] * b_size
        network_2_lst = [None] * b_size
        next_network_1_lst = [None] * b_size
        next_network_2_lst = [None] * b_size
        job_waiting_1_lst = [None] * b_size
        job_waiting_2_lst = [None] * b_size
        next_job_waiting_1_lst = [None] * b_size
        next_job_waiting_2_lst = [None] * b_size
        a_lst = [None] * b_size
        r_lst = [None] * b_size
        prob_a_lst = [None] * b_size
        sojourn_time_lst = [None] * b_size
        action_mask_lst = [None] * b_size
        subtask_index_lst = [None] * b_size

        for idx, transition in enumerate(self.data):
            #self.history.append(transition)
            network, job_waiting, a, r, nxt_network, nxt_job_waiting, prob_a, sojourn_time, action_mask, subtask_index = transition

            r_lst[idx] = [r*self.config["reward_weight"]]

            network_1_lst[idx] = network[0]
            network_2_lst[idx] = network[1]

            job_waiting_1_lst[idx] = job_waiting[0].tolist()
            job_waiting_2_lst[idx] = job_waiting[1].tolist()

            a_lst[idx] = [a]

            next_network_1_lst[idx] = nxt_network[0]
            next_network_2_lst[idx] = nxt_network[1]
            
            next_job_waiting_1_lst[idx] = nxt_job_waiting[0].tolist()
            next_job_waiting_2_lst[idx] = nxt_job_waiting[1].tolist()

            prob_a_lst[idx] = [prob_a]
            sojourn_time_lst[idx] = [sojourn_time * 5]

            action_mask_lst[idx] = action_mask
            subtask_index_lst[idx] = [subtask_index]


            if idx == self.config["batch_size"] - 1:
                self.data = self.data[self.config["batch_size"]:]
                break

        else:
            self.data = []
        

        # gnn sample을 배치단위로 inference하려면 이렇게 묶어줘야 함.
        network_1_loader = DataLoader(network_1_lst, batch_size=len(network_1_lst))
        next_network_1_loader = DataLoader(next_network_1_lst, batch_size=len(network_1_lst))
        network_1_batch = next(iter(network_1_loader))
        next_network_1_batch = next(iter(next_network_1_loader))

        network_2_loader = DataLoader(network_2_lst, batch_size=len(network_2_lst))
        next_network_2_loader = DataLoader(next_network_2_lst, batch_size=len(network_2_lst))
        network_2_batch = next(iter(network_2_loader))
        next_network_2_batch = next(iter(next_network_2_loader))

        # print(job_waiting_lst)
        
        # job_waiting = torch.tensor(np.array(job_waiting_lst), dtype=torch.float)
        job_waiting_1 = torch.squeeze(torch.tensor(np.array(job_waiting_1_lst), dtype=torch.float), dim=1)
        job_waiting_2 = torch.squeeze(torch.tensor(np.array(job_waiting_2_lst), dtype=torch.float), dim=1)

        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst, dtype=torch.float)
        sojourn_time = torch.Tensor(sojourn_time_lst)

        next_job_waiting_1 = torch.squeeze(torch.tensor(np.array(next_job_waiting_1_lst), dtype=torch.float), dim=1)
        next_job_waiting_2 = torch.squeeze(torch.tensor(np.array(next_job_waiting_2_lst), dtype=torch.float), dim=1)
        
        prob_a = torch.tensor(prob_a_lst, dtype=torch.float)
        action_mask_lst = torch.tensor(action_mask_lst)
        
        subtask_index_lst = torch.tensor(subtask_index_lst, dtype=torch.long)
        
        # self.data = []
        return [network_1_batch, network_2_batch], [job_waiting_1, job_waiting_2], a, r, [next_network_1_batch, next_network_2_batch], [next_job_waiting_1, next_job_waiting_2], prob_a, sojourn_time, action_mask_lst, subtask_index_lst, min(self.config["batch_size"], idx + 1)
    
    def train_net(self):
        if self.config["current_learning_time"] == 0:
            return
        self = self.cuda()
        self.train()
        
        pre_advantage = 0.0
        torch.cuda.empty_cache()

        network_batch, job_waiting, a, r, next_network_batch, next_job_waiting, prob_a, sojourn_time, action_mask, subtask_index, N_size = self.make_batch()
        
        network_batch_1 = network_batch[0].clone().cuda()
        network_batch_2 = network_batch[1].clone().cuda()

        network_batch = [network_batch_1, network_batch_2]

        job_waiting_1 = job_waiting[0].clone().cuda()
        job_waiting_2 = job_waiting[1].clone().cuda()

        job_waiting = [job_waiting_1, job_waiting_2]

        a = a.clone().cuda()
        r = r.clone().cuda()

        next_network_batch_1 = next_network_batch[0].clone().cuda()
        next_network_batch_2 = next_network_batch[1].clone().cuda()

        next_network_batch = [next_network_batch_1, next_network_batch_2]

        next_job_waiting_1 = next_job_waiting[0].clone().cuda()
        next_job_waiting_2 = next_job_waiting[1].clone().cuda()

        next_job_waiting = [next_job_waiting_1, next_job_waiting_2]

        prob_a = prob_a.clone().cuda()
        sojourn_time = sojourn_time.clone().cuda()
        gamma_gpu = torch.Tensor([self.config["gamma"]]).clone().cuda()
        action_mask = action_mask.clone().cuda()
        subtask_index = subtask_index.clone().cuda()

        for i in range(self.config["current_learning_time"]):
            
            # void
            next_state = self.gnn([next_network_batch, next_job_waiting])
            cur_state = self.gnn([network_batch, job_waiting])

            next_v = self.v(next_state)
            cur_v = self.v(cur_state)

            td_target = r + gamma_gpu * next_v
            
            delta = td_target - cur_v
            
            delta = delta.detach().to('cpu').numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma_gpu * self.config["lambda"] * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            #adad = [i[0].item() for i in advantage_lst]
            #print(len(adad))
            #print(adad)
            advantage_lst = torch.tensor(advantage_lst, dtype=torch.float).cuda()

            

            v_loss = F.smooth_l1_loss(cur_v , td_target.detach())

            self.optimizer.zero_grad()
            v_loss.backward()
            self.optimizer.step()
            #print("v_loss", v_loss)
            
            cur_state = self.gnn([network_batch, job_waiting])
            cur_state = cur_state.unsqueeze(1)
            new_state = cur_state.repeat(1, self.config["model_num"], 1)

            pi, current_entropy, outputs, hidden = self.pi([new_state, (torch.zeros(1, N_size, self.config["lstm_hidden_num"]).cuda(), torch.zeros(1, N_size, self.config["lstm_hidden_num"]).cuda())])


            current_entropy = self.entropy_normalize_weight * current_entropy

            # true가 valid action

            #outputs = outputs.clone().detach()

            indices = subtask_index.view(-1)
            indexed_current_entropy = current_entropy[torch.arange(N_size), indices]

            indexed_outputs = outputs[torch.arange(N_size), indices, :]
            indexed_outputs = indexed_outputs * action_mask

            exp_sum = torch.sum(torch.exp(indexed_outputs), dim=1) - torch.sum(action_mask.logical_not(), dim=1) # 0인 애들 빼줌
            exp_sum = exp_sum.view(-1, 1) # 전치행렬

            indexed_outputs_a = indexed_outputs.gather(1,a)

            pi_a = torch.exp(indexed_outputs_a) / exp_sum

            # pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.min(torch.tensor(88), torch.log(pi_a + 1e-9) - torch.log(prob_a + 1e-9)))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage_lst
            surr2 = torch.clamp(ratio, 1-self.config["eps_clip"], 1+self.config["eps_clip"]) * advantage_lst
            
            pi_loss = - torch.min(surr1, surr2).view(-1) - self.config["entropy_weight"] * indexed_current_entropy

            # total_loss = v_loss + pi_loss

            self.optimizer.zero_grad()
            pi_loss.mean().backward()
            self.optimizer.step()

    def train_net_history(self):

        self = self.cuda()
        self.train()
        # pre_advantage = 0.0
        if len(self.history) <= 0:
            return

        for i in range(self.config["history_learning_time"]):
            torch.cuda.empty_cache()

            temp_return = self.make_batch_history()

            if temp_return == -1:
                return

            network_batch, job_waiting, a, r, next_network_batch, next_job_waiting, prob_a, sojourn_time, action_mask, subtask_index, N_size = temp_return

            if len(network_batch) == 0:
                return
            
            network_batch_1 = network_batch[0].clone().cuda()
            network_batch_2 = network_batch[1].clone().cuda()

            network_batch = [network_batch_1, network_batch_2]

            job_waiting_1 = job_waiting[0].clone().cuda()
            job_waiting_2 = job_waiting[1].clone().cuda()

            job_waiting = [job_waiting_1, job_waiting_2]

            a = a.clone().cuda()
            r = r.clone().cuda()

            next_network_batch_1 = next_network_batch[0].clone().cuda()
            next_network_batch_2 = next_network_batch[1].clone().cuda()

            next_network_batch = [next_network_batch_1, next_network_batch_2]

            next_job_waiting_1 = next_job_waiting[0].clone().cuda()
            next_job_waiting_2 = next_job_waiting[1].clone().cuda()

            next_job_waiting = [next_job_waiting_1, next_job_waiting_2]

            prob_a = prob_a.clone().cuda()
            sojourn_time = sojourn_time.clone().cuda()
            gamma_gpu = torch.Tensor([self.config["gamma"]]).clone().cuda()
            action_mask = action_mask.clone().cuda()
            subtask_index = subtask_index.clone().cuda()

            next_state = self.gnn([next_network_batch, next_job_waiting])
            cur_state = self.gnn([network_batch, job_waiting])

            next_v = self.v(next_state)
            cur_v = self.v(cur_state)

            td_target = r + gamma_gpu * next_v

            delta = td_target - cur_v
            
            delta = delta.detach().to('cpu').numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma_gpu * self.config["lambda"] * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            #adad = [i[0].item() for i in advantage_lst]
            #print(len(adad))
            #print(adad)
            advantage_lst = torch.tensor(advantage_lst, dtype=torch.float).cuda()



            v_loss = F.smooth_l1_loss(cur_v , td_target.detach())

            self.optimizer.zero_grad()
            v_loss.mean().backward()
            self.optimizer.step()

            td_target = r + gamma_gpu * next_v
            delta = td_target - cur_v

            cur_state = self.gnn([network_batch, job_waiting])
            cur_state = cur_state.unsqueeze(1)
            new_state = cur_state.repeat(1, self.config["model_num"], 1)

        
            pi, current_entropy, outputs, hidden = self.pi([new_state, (torch.zeros(1, N_size, self.config["lstm_hidden_num"]).cuda(), torch.zeros(1, N_size, self.config["lstm_hidden_num"]).cuda())])


            current_entropy = self.entropy_normalize_weight * current_entropy

            # true가 valid action

            #outputs = outputs.clone().detach()

            indices = subtask_index.view(-1)
            indexed_current_entropy = current_entropy[torch.arange(N_size), indices]

            indexed_outputs = outputs[torch.arange(N_size), indices, :]
            indexed_outputs = indexed_outputs * action_mask

            exp_sum = torch.sum(torch.exp(indexed_outputs), dim=1) - torch.sum(action_mask.logical_not(), dim=1) # 0인 애들 빼줌
            exp_sum = exp_sum.view(-1, 1) # 전치행렬

            indexed_outputs_a = indexed_outputs.gather(1,a)

            pi_a = torch.exp(indexed_outputs_a) / exp_sum

            # pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.min(torch.tensor(88), torch.log(pi_a + 1e-9) - torch.log(prob_a + 1e-9)))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage_lst
            surr2 = torch.clamp(ratio, 1-self.config["eps_clip"], 1+self.config["eps_clip"]) * advantage_lst
            pi_loss = - torch.min(surr1, surr2).view(-1) - self.config["entropy_weight"] * indexed_current_entropy

            # print(pi_loss)

            self.optimizer.zero_grad()
            pi_loss.mean().backward()
            self.optimizer.step()