#!/usr/bin/env python
# Authors: Junior Costa de Jesus #

import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32
from scripts.environment_stage_1 import Env
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import gc
import torch.nn as nn
import math
from collections import deque
import copy
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#---Directory Path---#
dir = os.path.dirname(os.path.realpath(__file__))
#---Functions to make network updates---#
load_dir="/home/wendy/RLprojects/ros_ddpg/src/turtlebot3_ddpg/scripts/Models/stage2/ddpg/450a.pt"
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data*(1.0 - tau)+ param.data*tau)

def hard_update(target,source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

#---Ornstein-Uhlenbeck Noise for action---#


class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.2, desired_action_stddev=0.1, adaptation_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance >= self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)

#---Critic--#

EPS = 0.003
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1./np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v,v)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, 125)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        # self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        
        self.fa1 = nn.Linear(action_dim, 125)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        # self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())
        
        self.fca1 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fca1.weight)
        self.fca1.bias.data.fill_(0.01)
        # self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        
        self.fca2 = nn.Linear(250, 1)
        nn.init.xavier_uniform_(self.fca2.weight)
        self.fca2.bias.data.fill_(0.01)
        # self.fca2.weight.data.uniform_(-EPS, EPS)
        
    def forward(self, state, action):
        xs = torch.relu(self.fc1(state))
        xa = torch.relu(self.fa1(action))
        x = torch.cat((xs,xa), dim=xs.ndimension()-1)
        x = torch.relu(self.fca1(x))
        vs = self.fca2(x)
        return vs

#---Actor---#

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w):
        super(Actor, self).__init__()
        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        
        self.fa1 = nn.Linear(state_dim, 250)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        # self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())
        
        self.fa2 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fa2.weight)
        self.fa2.bias.data.fill_(0.01)
        # self.fa2.weight.data = fanin_init(self.fa2.weight.data.size())
        
        self.fa3 = nn.Linear(250, action_dim)
        nn.init.xavier_uniform_(self.fa3.weight)
        self.fa3.bias.data.fill_(0.01)
        # self.fa3.weight.data.uniform_(-EPS,EPS)
        
    def forward(self, state):
        x = torch.relu(self.fa1(state))
        x = torch.relu(self.fa2(x))
        action = self.fa3(x)
        if state.shape <= torch.Size([self.state_dim]):
            action[0] = torch.sigmoid(action[0])*self.action_limit_v
            action[1] = torch.tanh(action[1])*self.action_limit_w
        else:
            action[:,0] = torch.sigmoid(action[:,0])*self.action_limit_v
            action[:,1] = torch.tanh(action[:,1])*self.action_limit_w
        return action

#---Memory Buffer---#

class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0
        
    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        
        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])
        done_array = np.float32([array[4] for array in batch])
        
        return s_array, a_array, r_array, new_s_array, done_array
    
    def len(self):
        return self.len
    
    def add(self, s, a, r, new_s, done):
        transition = (s, a, r, new_s, done)
        self.len += 1 
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)

#---Where the train is made---#

BATCH_SIZE = 256
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.005
load_models=True

class Trainer:
    
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w, ram,delay_time=2,noise_std=0.2,noise_clip=0.5):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        #print('w',self.action_limit_w)
        self.ram = ram
        self.update_time = 0
        self.delay_time=delay_time
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.beta=0.001
        #self.iter = 0 
        
        
        self.actor1 = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w).to(device)
        self.target_actor1 = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w).to(device)
        self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), LEARNING_RATE)
        #actor2
        self.actor2 = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w).to(device)
        self.target_actor2 = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w).to(device)
        self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), LEARNING_RATE)
        self.actor1_copy= Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w).to(device)
        self.actor2_copy= Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w).to(device)
        #critic1
        self.critic1 = Critic(self.state_dim, self.action_dim).to(device)
        self.target_critic1 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), LEARNING_RATE)
        #critic2
        self.critic2 = Critic(self.state_dim, self.action_dim).to(device)
        self.target_critic2 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), LEARNING_RATE)
        
        if load_models:
                checkpoint = torch.load("/home/wendy/RLprojects/ros_ddpg/src/turtlebot3_ddpg/scripts/Models/stage2/ddpg/450a.pt")
                print(checkpoint.keys())
                print(checkpoint['epoch'])
                print(checkpoint)
                #load critic1
                self.target_critic1.load_state_dict(checkpoint['target_critic1'])
                self.critic1.load_state_dict(checkpoint['critic1'])
                self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
                #load actor1
                self.target_actor1.load_state_dict(checkpoint['target_actor1'])
                self.actor1.load_state_dict(checkpoint['actor1'])
                self.actor1_optimizer.load_state_dict(checkpoint['actor1_optimizer'])
                #load critic2
                self.target_critic2.load_state_dict(checkpoint['target_critic2'])
                self.critic2.load_state_dict(checkpoint['critic2'])
                self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
                #load actor2
                self.target_actor2.load_state_dict(checkpoint['target_actor2'])
                self.actor2.load_state_dict(checkpoint['actor2'])
                self.actor2_optimizer.load_state_dict(checkpoint['actor2_optimizer'])
                
                
                # self.start_epoch = checkpoint['epoch']
                # self.sigma=checkpoint["sigma"]
                # self.noise= OUNoise(action_dim,max_sigma=self.sigma)
                rospy.loginfo("loadmodel")
        else:
                hard_update(self.target_actor1, self.actor1)
                hard_update(self.target_critic1, self.critic1)
                hard_update(self.target_actor2, self.actor2)
                hard_update(self.target_critic2, self.critic2)
        
        
    def get_exploitation_action(self,state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        #print('actionploi', action)
        return action.data.numpy()
    def select_action(self, state):
        state = torch.from_numpy(state).to(device)
        action1 = self.actor1.forward(state).detach()
        action2 =  self.actor2.forward(state).detach()

        q1 = self.critic1(state, action1)
        q2 = self.critic2(state, action2)

        action = action1 if q1 >= q2 else action2

        return action.cpu().data.numpy().flatten()   
    def get_exploration_action(self, state):
        state = torch.from_numpy(state).to(device)
        action1 = self.actor1.forward(state).detach()
        action2 = self.actor2.forward(state).detach()
        q1 = self.critic1(state, action1)
        q2 = self.critic2(state, action2)
        if q1>=q2:
            flag ="q1"
        else:
            flag = "q2"
        new_action = action1.cpu() if q1 >= q2 else action2.cpu()
        new_action = new_action.data.numpy() #+ noise
        #noise = self.noise.sample()
        #print('noisea', noise)
        #noise[0] = noise[0]*self.action_limit_v
        #noise[1] = noise[1]*self.action_limit_w
        #print('noise', noise)

        #print('action_no', new_action)
        new_action[0] = np.clip(new_action[0], 0., ACTION_V_MAX)
        new_action[1] = np.clip(new_action[1], -ACTION_W_MAX, ACTION_W_MAX)
        return flag,new_action
    
    def get_exploration_para_action(self, state):
        
        state = torch.from_numpy(state).to(device)
        action1 = trainer.actor1_copy.forward(state).detach()
        action2 = trainer.actor2.forward(state).detach()
        q1 = self.critic1(state, action1)
        q2 = self.critic2(state, action2)
        if q1>=q2:
            flag ="q1"
        else:
            flag = "q2"
        new_action = action1.cpu() if q1 >= q2 else action2.cpu()
        new_action = new_action.data.numpy() #+ noise
        # N = copy.deepcopy(noise.get_noise(t=step))
        # N[0] = N[0]*ACTION_V_MAX/2
        # N[1] = N[1]*ACTION_W_MAX
        # new_action[0] = np.clip(action[0] + N[0], 0., ACTION_V_MAX)
        # new_action[1] = np.clip(action[1] + N[1], -ACTION_W_MAX, ACTION_W_MAX)
        new_action[0] = np.clip(new_action[0], 0., ACTION_V_MAX)
        new_action[1] = np.clip(new_action[1], -ACTION_W_MAX, ACTION_W_MAX)
        
        
        return flag,new_action
           
    def add_noise_to_parameters(sefl,actor_copy,actor):
            hard_update(actor_copy,actor)
            parameters = actor_copy.state_dict()
            for name in parameters:
                parameter = parameters[name]
                
                rand_number = torch.randn(parameter.shape)

                # parameter = parameter + rand_number * parameter_noise.current_stddev
                # parameters[name]=parameter
                rand_number = rand_number.to(parameter.device)
                
                parameters[name]+=(rand_number * parameter_noise.current_stddev)
                parameters[name].to(dtype=torch.float64)
            actor_copy.cpu().load_state_dict(parameters)
            actor_copy.to(device)
    def update_diff(self,ram,actor,parameter_noise):
            
            noise_data_list = list(ram.buffer)
            
            noise_data_list = np.array(noise_data_list[-step_cntr:],dtype=object)
            actor_copy_state, actor_copy_action,reward, next_state, done= zip(*noise_data_list)

            #Noisetoi actoriin action
            # actor_copy_actions = np.array(actor_copy_action.numpy())
            # actor_copy_actions = np.vstack(actor_copy_action)
            actor_copy_actions  = np.vstack([t for t in actor_copy_action])

            #Engiin actoriin action
            actor_actions = []
            for state in np.array(actor_copy_state):
                state = Variable(torch.from_numpy(state).to(device, dtype=torch.float))
                action = actor.forward(state).cpu().detach().numpy()
                actor_actions.append(action)

            #Distance tootsoh
            diff_actions = actor_copy_actions - actor_actions        
            mean_diff_actions = np.mean(np.square(diff_actions),axis=0)
            distance = math.sqrt(np.mean(mean_diff_actions))
            #Sigma-g update hiih
            parameter_noise.adapt(distance)
    def softmax_operator(self, q_vals, noise_pdf=None):
        max_q_vals = torch.max(q_vals, 1, keepdim=True).values
        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = torch.exp(self.beta * norm_q_vals)
        Q_mult_e = q_vals * e_beta_normQ

        numerators = Q_mult_e
        denominators = e_beta_normQ

        if self.with_importance_sampling:
            numerators /= noise_pdf
            denominators /= noise_pdf

        sum_numerators = torch.sum(numerators, 1)
        sum_denominators = torch.sum(denominators, 1)

        softmax_q_vals = sum_numerators / sum_denominators

        softmax_q_vals = torch.unsqueeze(softmax_q_vals, 1)
        return softmax_q_vals
    
    def optimizer(self,update_q1):
        s_sample, a_sample, r_sample, new_s_sample, done_sample = ram.sample(BATCH_SIZE)
        
        s_sample = torch.from_numpy(s_sample).to(device)
        a_sample = torch.from_numpy(a_sample).to(device)
        r_sample = torch.from_numpy(r_sample)
        new_s_sample = torch.from_numpy(new_s_sample).to(device)
        done_sample = torch.from_numpy(done_sample)
        
        with torch.no_grad():
            if update_q1:
               
                a_target = self.target_actor1.forward(new_s_sample).detach()
            else:
                a_target = self.target_actor2.forward(new_s_sample).detach()
        
        
        #-------------- optimize critic
        
            noise = torch.randn_like(torch.zeros(a_target.size())).to(device) * self.noise_std
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            a_target  += noise
            # N = copy.deepcopy(noise.get_noise(t=step))
            # N[0] = N[0]*ACTION_V_MAX/2
            # N[1] = N[1]*ACTION_W_MAX
            # new_action[0] = np.clip(action[0] + N[0], 0., ACTION_V_MAX)
            # new_action[1] = np.clip(action[1] + N[1], -ACTION_W_MAX, ACTION_W_MAX)
            a_target[0] = torch.clip(a_target[0], 0., ACTION_V_MAX)
            a_target[1] = torch.clip(a_target[1], -ACTION_W_MAX, ACTION_W_MAX)
            c1_next_value = torch.squeeze(self.target_critic1.forward(new_s_sample, a_target).detach().cpu())
            c2_next_value = torch.squeeze(self.target_critic2.forward(new_s_sample, a_target).detach().cpu())
            critic_val=torch.min(c1_next_value,c2_next_value)
            # critic_val = self.softmax_operator(critic_val)
            # y_exp = r _ gamma*Q'(s', P'(s'))
            y_expected = r_sample + (1 - done_sample)*GAMMA*critic_val
            # y_pred = Q(s,a)
            
           
            #-------Publisher of Vs------
            # self.qvalue = y_predicted.detach()
            # self.pub_qvalue.publish(torch.max(self.qvalue))
            #print(self.qvalue, torch.max(self.qvalue))
            #----------------------------
            
        if update_q1:
            y_predicted_1 = torch.squeeze(self.critic1.forward(s_sample, a_sample)).cpu()

            critic1_loss =F.smooth_l1_loss(y_predicted_1, y_expected).to(device)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()
            
            
            pred_a_sample = self.actor1.forward(s_sample)
            average_critic=self.critic1.forward(s_sample, pred_a_sample)
            loss_actor1 = -1*torch.sum(average_critic).to(device)
            self.actor1_optimizer.zero_grad()
            loss_actor1.backward()
            self.actor1_optimizer.step()
            soft_update(self.target_actor1, self.actor1, TAU)
            soft_update(self.target_critic1, self.critic1, TAU)


        else:
            y_predicted_2 = torch.squeeze(self.critic2.forward(s_sample, a_sample)).cpu()

            critic2_loss =F.smooth_l1_loss(y_predicted_2, y_expected).to(device)

			
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

             
            pred_a_sample = self.actor2.forward(s_sample)
            average_critic=self.critic2.forward(s_sample, pred_a_sample)
            loss_actor2 = -1*torch.sum(average_critic).to(device)
            self.actor2_optimizer.zero_grad()
            loss_actor2.backward()
            self.actor2_optimizer.step()
            soft_update(self.target_actor2, self.actor2, TAU)
            soft_update(self.target_critic2, self.critic2, TAU)
    
    # def save_models(self, episode_count):

    #     torch.save(self.target_actor.state_dict(), dirPath +'/Models/' + world + '/' + str(episode_count)+ '_actor.pt')
    #     torch.save(self.target_critic.state_dict(), dirPath + '/Models/' + world + '/'+str(episode_count)+ '_critic.pt')
    #     print('****Models saved***')
    def save_model(self,dir,ep,parameter1_noise,parameter2_noise,rewards,total_sum_count,total_success_count,method):
        state = {'target_critic1':trainer.target_critic1.state_dict(),'critic1':trainer.critic1.state_dict(), 'critic1_optimizer':trainer.critic1_optimizer.state_dict(), 
                 'target_critic2':trainer.target_critic2.state_dict(),'critic2':trainer.critic2.state_dict(), 'critic2_optimizer':trainer.critic2_optimizer.state_dict(), 
                 'target_actor1':trainer.target_actor1.state_dict(),'actor1':trainer.actor1.state_dict(), 'actor1_optimizer':trainer.actor1_optimizer.state_dict(), 
                 'target_actor2':trainer.target_actor2.state_dict(),'actor2':trainer.actor2.state_dict(), 'actor2_optimizer':trainer.actor2_optimizer.state_dict(), 
                 'epoch':ep,'parameter1_noise.current_stddev':parameter1_noise.current_stddev,'parameter2_noise.current_stddev':parameter2_noise.current_stddev,
                 'rewards':rewards,'total_sum_count':total_sum_count,'total_success_count':total_success_count}
        torch.save(state,dir + '/Models/' + world + '/'+method+ '/'+str(ep)+"a.pt")
        
        
    def load_models(self, episode):
        self.actor.load_state_dict(torch.load(dir + '/Models/' + world + '/'+str(episode)+ '_actor.pt'))
        self.critic.load_state_dict(torch.load(dir + '/Models/' + world + '/'+str(episode)+ '_critic.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print('***Models load***')

#---Mish Activation Function---#
def mish(x):
    '''
        Mish: A Self Regularized Non-Monotonic Neural Activation Function
        https://arxiv.org/abs/1908.08681v1
        implemented for PyTorch / FastAI by lessw2020
        https://github.com/lessw2020/mish
        param:
            x: output of a layer of a neural network
        return: mish activation function
    '''
    return x*(torch.tanh(F.softplus(x)))

#---Run agent---#

is_training = True

exploration_decay_rate = 0.001

MAX_EPISODES = 10001
MAX_STEPS = 300
MAX_BUFFER = 200000
rewards_all_episodes = []

STATE_DIMENSION = 14
ACTION_DIMENSION = 2
ACTION_V_MAX = 0.22 # m/s
ACTION_W_MAX = 2. # rad/s
world = 'stage2'

if is_training:
    var_v = ACTION_V_MAX*.5
    var_w = ACTION_W_MAX*2*.5
else:
    var_v = ACTION_V_MAX*0.10
    var_w = ACTION_W_MAX*0.10

print('State Dimensions: ' + str(STATE_DIMENSION))
print('Action Dimensions: ' + str(ACTION_DIMENSION))
print('Action Max: ' + str(ACTION_V_MAX) + ' m/s and ' + str(ACTION_W_MAX) + ' rad/s')
ram =  MemoryBuffer(MAX_BUFFER)
ram1 =  MemoryBuffer(MAX_BUFFER)
ram2 =  MemoryBuffer(MAX_BUFFER)
trainer = Trainer(STATE_DIMENSION, ACTION_DIMENSION, ACTION_V_MAX, ACTION_W_MAX, ram,2)

parameter_noise = AdaptiveParamNoiseSpec(desired_action_stddev=0.2)
#trainer.load_models(4880)




if __name__ == '__main__':
    rospy.init_node('ddpg_depict_gpu_parameter_noisy_optimal_2actors')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env(action_dim=ACTION_DIMENSION)
    before_training = 4

    past_action = np.zeros(ACTION_DIMENSION)
    stage_name="stage2"
    total_sum_count=0
    total_success_count=0
    method ="ddpg"
    
    start_epoch =0
    past_action = np.zeros(ACTION_DIMENSION)
    start_time =time.time()
    rewards=[]
    writer = SummaryWriter("/home/wendy/RLprojects/ros_ddpg/src/turtlebot3_ddpg/logs/ddpg")
    parameter_initial_stddev=0.1
    if  load_models:
        checkpoint = torch.load(load_dir)
        rewards=checkpoint['rewards']
        total_sum_count=checkpoint['total_sum_count']
        total_success_count=checkpoint['total_success_count']
        start_epoch=checkpoint['epoch']
        para_curren_stddev=checkpoint['epoch']
        parameter1_initial_stddev=checkpoint['parameter1_noise.current_stddev']
        parameter1_noise = AdaptiveParamNoiseSpec(initial_stddev=parameter1_initial_stddev,desired_action_stddev=0.2)
        parameter2_initial_stddev=checkpoint['parameter2_noise.current_stddev']
        parameter2_noise = AdaptiveParamNoiseSpec(initial_stddev=parameter2_initial_stddev,desired_action_stddev=0.2)
        rospy.loginfo("loadmodel")  
    else:
        parameter1_noise = AdaptiveParamNoiseSpec(desired_action_stddev=0.2)
        parameter2_noise = AdaptiveParamNoiseSpec(desired_action_stddev=0.2)
       
    for ep in range(start_epoch+1,MAX_EPISODES):
        done = False
        state = env.reset()
        # if is_training and not ep%10 == 0 and ram.len >= before_training*MAX_STEPS:
        #     print('---------------------------------')
        #     print('Episode: ' + str(ep) + ' training')
        #     print('---------------------------------')
        # else:
        #     if ram.len >= before_training*MAX_STEPS:
        #         print('---------------------------------')
        #         print('Episode: ' + str(ep) + ' evaluating')
        #         print('---------------------------------')
        #     else:
        #         print('---------------------------------')
        #         print('Episode: ' + str(ep) + ' adding to memory')
        #         print('---------------------------------')

        rewards_current_episode = 0.
        step_cntr = 0
        
        trainer.add_noise_to_parameters(trainer.actor1_copy,trainer.actor1)
     

        
        for step in range(MAX_STEPS):
            state = np.float32(state)

            if is_training and not ep%10 == 0 and ram.len >= before_training*MAX_STEPS:
            # if is_training:
               flag, action = trainer.get_exploration_para_action(state)
              
            else:
               flag, action = trainer.get_exploration_action(state)
            # action = trainer.get_exploration_para_action(state)
            if not is_training:
                action = trainer.get_exploitation_action(state)
            next_state, reward, done,goal = env.step(action, past_action)
            # print('action', action,'r',reward)
            past_action = copy.deepcopy(action)
            
            rewards_current_episode += reward
            next_state = np.float32(next_state)
           
            # if not ep%10 == 0 or not ram.len >= before_training*MAX_STEPS:
                
            #   if reward == 1000.:
            #         print('***\n-------- Maximum Reward ----------\n****')
            #         for _ in range(3):
            #             ram.add(state, action, reward, next_state, done)
            #   else:
            #         ram.add(state, action, reward, next_state, done)
            
            ram.add(state, action, reward, next_state, done)
            if flag =="q1":
                ram1.add(state, action, reward, next_state, done)
            else:
                ram2.add(state, action, reward, next_state, done)
            state = copy.deepcopy(next_state)
            
            if (done or goal):   
                # sum_count_episode=sum_count_episode+1
                total_sum_count=total_sum_count+1
                if goal:
                #   success_count_episode=success_count_episode+1
                    total_success_count=total_success_count+1
            

            if ram.len >= BATCH_SIZE:

                trainer.optimizer(True)
                trainer.optimizer(False)
            

            if done or step == MAX_STEPS-1: 
                if(step == MAX_STEPS-1):
                   rospy.loginfo("timeout")
                if(total_sum_count==0):
                    total_success_rate=0
                else:
                    total_success_rate=total_success_count/total_sum_count
                m,s =divmod(int(time.time()- start_time),60)
                h,m =divmod(m,60)
                rewards.append(rewards_current_episode)
                # success_rates.append(success_rate_episode)
                writer.add_scalar("Total reward per episode_"+stage_name,rewards_current_episode,ep)
                rospy.loginfo('回合: %d 得分: %.2f 记忆量: %d actor1_para_noise_stddev: %.2f actor2_para_noise_stddev: %.2f 花费时间: %d:%02d:%02d,总成功率：%.3f',ep, rewards_current_episode, ram.len, parameter1_noise.current_stddev, parameter2_noise.current_stddev,h, m, s,total_success_rate)
        
                break
            step_cntr += 1 
        average_reward = np.average(rewards)
        writer.add_scalar("average Reward:"+stage_name,average_reward,ep)
        writer.add_scalar("total succes rate:"+stage_name,total_success_rate,ep) 
        
        if ram1.len>0:
         trainer.update_diff(ram1,trainer.actor1,parameter1_noise)
               
       
            
            
        if ep%10 == 0:
            trainer.save_model(dir,ep,parameter1_noise,parameter2_noise,rewards,total_sum_count,total_success_count,method)


print('Completed Training')