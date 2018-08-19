# AI of Self Driving Car with 3 Signals


# Importing the library

import numpy as np               # Number format
import random                    # during Experience replay
import os                        # Loading the model , save the brain of ai
import torch                     # handle time and graphs better for Ai development
import torch.nn as nn                  # contian signals contribution or q values nn-> neural network
import torch.nn.functional as F      # contain diff function like lose function, softmax
import torch.optim as optim       # optimizer for gradient process
import torch.autograd as autograd         # convertor from tensor to variable as gradient
from torch.autograd import Variable       # import variable as gradient


# Creating the Architecture of Neural Network
#  AI is class we create an  object

class Network(nn.module):                  # to inherited all tools of nn module
    
    def __init__(self, input_size, nb_action):                 # Self is complusory as referring to object 2nd parameter is no of input neurons(our case 5) , 3rd is output neurons(our case 3)
        super(Network, self).__init__()                  ## use tools of nn module
        self.input_size = input_size                   # no. of input neurons
        self.nb_action = nb_action                # no of output variable
        self.fc1 = nn.Linear(input_size , 30)                    # Full connection b/w different layers Linear has input_features , out_features(hidden layer for this case) , bias is true by default
        self.fc2 = nn.Linear(30, nb_action)                # Full connection b/w hidden layer and output Layers
                        
    def forward(self , state):               # Forward Function activate neuron, state neurons entering and this function also return Q values for each action
       x =  F.relu(self.fc1(state))               # Activate hidden neurons use function torch nn relu functional
                # Q values for each action output neurons as fc2 (x) is hidden neurons
       return self.fc2(x)             # returing each action left, rigth , staight
   
## Implementing Experience Replay
       
class ReplayMemory(object):
    
    def __init__(self, capacity):          # capacity denotes size  last 100 or more transitions state
        self.capacity = capacity            # self capacity is related to object
        self.memory = []                    # store each transitions
        
    def push(self, event):                  # event is transition added to memory 4 elements (last state , new state ,last action , last rewards)
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        # Ex list = ((1,2,3), (4,5,6)) , then zip(*list) = ((1,4), (2,3), (5,6))
        samples = zip(*random.sample(self.memory, batch_size)) # random sample from memory zip is reshape funtion  each action in one batch
        return map(lambda x: Variable(torch.cat(x , 0)) , samples)                 # In order to return sample convert into pytorch first using map Concatenate each batch in sample as one dimensional using torch 

## Implementing deep Q Learning
    
class Dqn():                    # Brain of AI
    
    def __init__(self ,input_size, nb_action, gamma):  # Gamma value used to calculate cost function hidden layers(input_size) output action and other var used in object used
        self.gamma = gamma                              # Object gamma initialize
        self.reward_window = []                      # Mean of reward present in memory
        self.model = Network(input_size, nb_action)    # call for Network class object of model
        self.memory = ReplayMemory(100000)                    # memory object of ReplayMemory class(arg is capacity(no. transactions)) as var of AI class 
        self.optimizer =  optim.Adam(self.model.parameters(), lr = 0.001)                           # tools(gradient methods) required is in torch.optimizers(class) lr denotes learning rate(not to fast)
        self.last_state =   torch.Tensor(input_size).unsqueeze(0)                          # last state(input batch) is vector(torch tenser object) in dimension(input_size) also contain one fake vector(unsqueeze first dimension) that correspond to batch process(signal orientaion and -ve orientation) 0->fake dimension of state
        self.last_action =   0                                  # Action is 0(index 0) , 20(index 1), -20(index 2) Initalize it by 0 (index 0)
        self.last_reward = 0                    # Initial last reward of car is 0
        
        
    def select_action(self, state):              # Select action left , right  as depend on input state return  output  as Q value
        probs = F.softmax(self.model(Variable(state, volatile = True))*7)         # produce prob of each action(three action from input state) Q1 , Q2, Q3 using softmax function whose input is Q value(Q1 , Q2 ,Q3) of input state and then rap the input tensor into torch variable state then * by temperature(high) as nn to sure(high) which action to play
        # softmax([1 ,2 , 3]) = [0.4, 0.11, 0.85] => softmax([1,2,3]*3) = [0, 0.02, 0.98] as temperature is 3
        action =  probs.multinomial()                  # Random draw of prob distribution
        return    action.data[0,0]                  # action store fake batch at 0,0

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):        # Base of deep Q learning Markov Process. transition of batch on samples for each state(store reward , next state...)
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)        # Require output of action decided by network the best action from batch_action, unsqueeze 1 -> fake action , batch_state & batch_action of same dimension , squeeze to convert batch(fake dimension) to tensor var
        next_outputs = self.model(batch_next_state).detach().max(1)[0]          # compute the next output to get the target as reward + gamma * next_output  , among all next batch output detach max Q value of action(index 1) from fake batch state dimension(index 0)
        target  = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)                                                      # temproral difference loss using predefined function
        self.optimizer.zero_grad()                                    # apply loss error to perform gradient and update weight, reinialize the optimizer
        td_loss.backward(retain_variables = True)                 # updated weight back propagated to the network (backpropagation) retain var to free memory as we iterate several times
        self.optimizer.step()                                       # weight updated
        
        
    def update(self, reward, new_signal):                     # when AI discovered new state and recieve reward depending of the action & than update all elements in transaction function this fun used in map.py to update last signal , reward
        new_state =  torch.Tensor(new_signal).float().unsqueeze(0)                           # new signal is next state of car we need to covert into torch tensor (float) and then unsqueeze
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))  # As all element is torch tensor form                                                             # we need to update memory as update next batch state so used push function in replay memory class
        action = self.select_action(new_state)                    # Start action again 
        if len(self.memory.memory) > 100:                                     # First memory is object and second memory is array
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state                                             # Update all variable
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window[0]) > 1000:
            del self.reward_window[0]
        return action   
       
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)                  # len is not equal to zero
    
    def save(self):
        torch.save({'state_dict':self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('=> loading checkpoint.....')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('done !')
        else:
           print('no checkpoint found')