

import random
import numpy as np
import collections
from environment_stage_1_her import Env
class Trajectory:
    ''' 用来记录一条完整轨迹 '''
    def __init__(self, init_state):
        self.states = [init_state]
        self.next_states=[]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.length = 0

    def store_step(self, state, action, reward, next_state, done):
        self.actions.append(action)
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length += 1


class ReplayBuffer_Trajectory:
    ''' 存储轨迹的经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.env = Env(action_dim=2)

    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size, use_her, dis_threshold=0.15, min_range=0.136, her_ratio=0.8):
        batch = dict(states=[],
                     actions=[],
                     next_states=[],
                     rewards=[],
                     dones=[])
        for _ in range(batch_size):
            traj = random.sample(self.buffer, 1)[0]
            step_state = np.random.randint(traj.length)
            state = traj.states[step_state]
            next_state = traj.next_states[step_state]
            action = traj.actions[step_state]
            reward = traj.rewards[step_state]
            done = traj.dones[step_state]

            if use_her and np.random.uniform() <= her_ratio:
                step_goal = np.random.randint(step_state + 1, traj.length + 1)
                goal = traj.states[step_goal][:2]  # 使用HER算法的future方案设置目标
                # state = self.env.getState()
                
                dis = np.sqrt(np.sum(np.square(next_state[:2] - goal)))
                
                if dis <= dis_threshold:
                   reward = 1000 
                elif min_range > min(next_state[2:12]) > 0:
                   reward = -500
                   done = True
                else:
                    reward=-1
        
                state = np.hstack((state[:-2], goal))
                next_state = np.hstack((next_state[:-2], goal))
                # reward,done = self.env.setReward(state,False)

            batch['states'].append(state)
            batch['next_states'].append(next_state)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['dones'].append(done)

        batch['states'] = np.array(batch['states'])
        batch['next_states'] = np.array(batch['next_states'])
        batch['actions'] = np.array(batch['actions'])
        batch['rewards'] =np.array(batch['rewards'])
        batch['dones'] =np.array(batch['dones'])
        return batch