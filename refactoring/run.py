import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import random
import time
from time import strftime, localtime
import json
from collections import deque

from Agent import PPO

def run(config, env, manager):
    
    model = PPO(config)
    model.print_parameter_number()

    if config.load_model:
        model.load_state_dict(torch.load(config.model_path))

    max_reward = 0

    average_reward = 0

    temp_history = deque([])
    episode_history = []
    actions = []

    void_mask = [0] * config["node_num"] + [1]
    unvoid_mask = [1] * config["node_num"] + [0]

    episode_total_reward = 0
    hidden = (torch.zeros(1, 1, config["lstm_hidden_num"]), torch.zeros(1, 1, config["lstm_hidden_num"]))
    
    while True:
        # time.sleep(config["cpu_load_balance_time"])
        
        model = model.to('cpu')
        msg = env.return_env_status()
        
        if msg == "action": # omnet의 메세지, state 받으면 됨
            
            cur_state = env.return_state()

            manager.record_summary("completeJobNum/train", cur_state["cur_network"]["complete_job_num"], env.get_time_step())
            manager.record_summary("reward/train", cur_state["cur_network"]["reward"], env.get_time_step())
            manager.update_episode_total_reward(cur_state["cur_network"]["reward"])

            
            if config.is_train: # 보류
                if temp_history:
                    # temp_history[-1][3] = reward_2
                    temp_history[-1][4] = network_state
                    temp_history[-1][5] = job_waiting_state

                while temp_history:
                    history = temp_history.popleft()
                    
                    # model.history.append(history)
                    episode_history.append(history)
                    model.put_data(history)

            network_state = [cur_state["pre_network"]["network_state"], cur_state["cur_network"]["network_state"]]
            job_waiting_state = [cur_state["pre_network"]["job_waiting_state"], cur_state["cur_network"]["job_waiting_state"]]

            temp_history = deque([])


            job_waiting_num = 0
            job_waiting_queue = deque()
            for job in cur_state["cur_network"]["job_waiting_state"]:
                if any(job): # 하나라도 0이 아닌 것 이 있으면 job이 있는것임.
                    job_waiting_num += 1
                    job_waiting_queue.append(job)
                
            if env.get_time_step() > 1:
                if cur_state["cur_network"]["reward"] != 0:
                    with torch.no_grad():
                        state = model.gnn([network_state, job_waiting_state])
                        manager.record_summary("Value/train", torch.mean(model.v(state)), env.get_time_step())

            job = job_waiting_queue.popleft()
            src = -1
            dst = 1
            for i in range(config.node_num):
                if job[i] == -1:
                    src = i
                if job[i] == 1:
                    dst = i

            if src == -1:
                src = dst
                
            #print(f"src : {src}, dst : {dst}")
            #print(job)
            subtasks = job[config.node_num:]
            offloading_vector = []

            

            with torch.no_grad():
                feature = model.gnn([network_state, job_waiting_state])
                feature = F.normalize(feature, dim=1)
                new_feature = feature.unsqueeze(0)
                first_prob, entropy, output, hidden = model.pi([new_feature, hidden])

            manager.record_summary("Entropy/train", torch.mean(entropy).item(), env.get_time_step())


            m = Categorical(first_prob[0][0]) # 첫 번째 batch의 첫 번째 node+void개의 확률들
            nodes = m.sample()

            node = nodes.item()

            #print(f'node : {node}')
            
            # void action 실험용
            # node = nodeNum 
            
            
            # void action 뽑으면 void만 업데이트
            if node == config["node_num"]: 
                actions.append(node)
                action_mask = void_mask

                temp_history.append([
                    [cur_state["pre_network"]["network_state"], cur_state["cur_network"]["network_state"]], 
                    [cur_state["pre_network"]["job_waiting_state"], cur_state["cur_network"]["job_waiting_state"]], 
                    node, 0, 
                    [cur_state["pre_network"]["network_state"], cur_state["cur_network"]["network_state"]], 
                    [cur_state["pre_network"]["job_waiting_state"], cur_state["cur_network"]["job_waiting_state"]], 
                    1, 
                    0, action_mask, 0]
                )

                env.step("void")

                #print("action finish.")
                
                if env.get_response() == "ok":
                    manager.update_void_selected_num()

            else: # void action 아니면 다른 action 시작
                
                if random.random() > config["imitation_probability"]:
                        config["our"] = True
                else:
                    config["our"] = False
                

                for sub_index in range(config.model_num):

                    with torch.no_grad():
                        feature = model.gnn([network_state, job_waiting_state])
                        feature = F.normalize(feature, dim=1)
                        new_feature = feature.unsqueeze(0)
                        prob, entropy, output, hidden = model.pi([new_feature, hidden])
                        # print(prob)

                    manager.record_summary("Entropy/train", torch.mean(entropy).item(), env.get_time_step())

                    output = output[:, :, 0:-1].squeeze(0) # action중 마지막 action은 masking

                    prob = F.softmax(output, dim=1)
                    prob = torch.concat([prob, torch.zeros(1, 1)], dim=1)

                    m = Categorical(prob)
                    nodes = m.sample()
                    action_mask = unvoid_mask


                    if config["our"]:
                        node = nodes[0].item() # 확률에서 sampling
                    else:
                        node = torch.argmin(cur_state["cur_network"]["node_waiting_state"][:,0]).item() # argmin으로 가장 waiting이 작은 node 뽑음

                    next_state = env.return_estimated_state(cur_state, node, subtasks, sub_index) # 예측된 state를 반환

                    temp_history.append([
                    [cur_state["pre_network"]["network_state"], cur_state["cur_network"]["network_state"]], 
                    [cur_state["pre_network"]["job_waiting_state"], cur_state["cur_network"]["job_waiting_state"]], 
                    node, 0, 
                    [next_state["pre_network"]["network_state"], next_state["cur_network"]["network_state"]], 
                    [next_state["pre_network"]["job_waiting_state"], next_state["cur_network"]["job_waiting_state"]], 
                    prob[0][node].item(),
                    0, action_mask, 0]
                    )

                    offloading_vector.append(node)
                    manager.update_node_selected_num(node)

                    actions.append(node)

                    cur_state = next_state # 현재 state를 next로 변경

                env.set_time_step(env.get_time_step() + 1) # step + 1

                if len(offloading_vector) != 0: # for문을 다 돌면 -> void action 안뽑으면
                    # print(offloading_vector)
                    action = str(offloading_vector)

                    env.step(action)

        elif msg == "stop":
            env.send_response()
            
            
        elif msg == "episode_finish":
            env.send_response()

            env.set_episode(env.get_episode() + 1) # episode + 1


            actions = []

            manager.record_node_selected_num(env.get_episode())
            manager.record_void_selected_num(env.get_episode())

            episodic_reward = env.get_episode_result()
            episodic_reward = json.loads(episodic_reward)

            env.send_response()
            
            finish_num = float(episodic_reward['reward'])
            complete_num = int(episodic_reward['completNum'])
            average_latency = float(episodic_reward['averageLatency'])
            jitter = episodic_reward['jitter']
            jitterMake = episodic_reward['jitterMake']
            action_id = episodic_reward['action_id']
            action_reward = episodic_reward['action_reward']
            #print(list(map(float, jitter.strip().split(" "))))
            #print(list(map(float, jitterMake.strip().split(" "))))
            #print(list(map(int, action_id.strip().split(" "))))
            #print(list(map(float, action_reward.strip().split(" "))))


            normalized_finish_num = model.return_normalize_reward(finish_num)
            
            manager.record_summary("EpisodicReward/train", finish_num, env.get_episode())
            manager.record_summary("NormalizedEpisodicReward/train", normalized_finish_num, env.get_episode())
            manager.record_summary("CompleteNum/train", complete_num, env.get_episode())
            manager.record_summary("averageLatency/train", average_latency ,env.get_episode())
            manager.record_episode_total_reward(env.get_episode())
            
            model.data = episode_history[:]
            data_length = len(model.data)
            reward_list = [(data_length - i) * -0.01 for i in range(data_length)]

            action_id_list = list(map(int, action_id.strip().split(" "))) if action_id != '' else []
            action_reward_list = list(map(float, action_reward.strip().split(" "))) if action_reward != '' else []

            action_id_reward_list = sorted(zip(action_id_list, action_reward_list))

            #print(action_id_reward_list)

            for i in range(len(action_id_reward_list)):
                index = action_id_reward_list[i][0]
                if index < data_length:
                    reward_list[index] = action_id_reward_list[i][1]

            #print(reward_list)

            for i in range(len(model.data)):
                model.data[i][3] = reward_list[i]

            tt = []
            for i in range(len(model.data)):
                tt.append(model.data[i][3])
            #print(tt)
            #print(len(model.data))
            #print(tt)
            episode_history = []
            temp_history = deque([])

            

            hidden = (torch.zeros(1, 1, config["lstm_hidden_num"]), torch.zeros(1, 1, config["lstm_hidden_num"]))

            

            config["entropy_weight"] = max(0.0001, config["entropy_weight"] * config["entropy_gamma"])
            config["imitation_probability"] = max(config["imitation_gamma"] * config["imitation_probability"], 0.2)

            if finish_num > max_reward:
                modelPathName = config["path_name"] + "/max_model.pth"
                torch.save(model.state_dict(), modelPathName)
                max_reward = finish_num

            manager.record_summary("AverageReward/train", average_reward, env.get_episode())
            average_reward = 0

            if config["is_train"]:
                
                if env.get_episode() % 100 == 0:
                    tm = localtime(time.time())
                    time_string = strftime('%Y-%m-%d %I:%M:%S %p', tm)
                    print(f"[{time_string}] training....")
                model.train_net()
                if env.get_episode() % 100 == 0:
                    tm = localtime(time.time())
                    time_string = strftime('%Y-%m-%d %I:%M:%S %p', tm)
                    print(f"[{time_string}] training complete")

                if env.get_episode() % 100 == 0:
                    tm = localtime(time.time())
                    time_string = strftime('%Y-%m-%d %I:%M:%S %p', tm)
                    print(f"[{time_string}] training replay buffer....")
                model.train_net_history()
                if env.get_episode() % 100 == 0:
                    tm = localtime(time.time())
                    time_string = strftime('%Y-%m-%d %I:%M:%S %p', tm)
                    print(f"[{time_string}] training complete")

                model.clear_data()

                if env.get_episode() % 100 == 0:
                    modelPathName = config["path_name"] + "/model.pth"
                    torch.save(model.state_dict(), modelPathName)
                    modelPathName = config["path_name"] + f"/model_{env.get_episode()}.pth"
                    torch.save(model.state_dict(), modelPathName)

                    time.sleep(10)

                model.eval()