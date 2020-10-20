import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
from collections import deque
import random
import pickle
from PER import *
import matplotlib.pyplot as plt 
import itertools 

# %%
#########################
###    ENVIRONMENT    ###
#########################
class PBN_env:
    #Create the environment
    N_NODES = 28
    N_INPUTS = 3
    
    if N_INPUTS >=1:
        POSSIBLE_P = ["".join(seq) for seq in itertools.product("10", repeat=N_INPUTS)]
        SIZE_POSSIBLE_P = len(POSSIBLE_P)
    else:
        POSSIBLE_P = 1
        SIZE_POSSIBLE_P = 1

    def reset(self):
        self.state = list(np.random.randint(0, 2, self.N_NODES))
        return self.state

    def action_space(self):
        return len(self.POSSIBLE_P)
    
    def computeStateNbr(self, x):
        if type(x) == str:
            stateNbr = np.abs(np.power(2, env.N_NODES)-1-int(x,2))
        else:
            stateStr = ''.join(str(s) for s in x)
            stateNbr = np.abs(np.power(2, env.N_NODES)-1-int(stateStr,2))
        return stateNbr
    
    def computeStateStr(self, x):
        if type(x) == list:
            stateStr = ''.join(str(s) for s in x)
        else:
            stateStr = bin(np.abs((x-np.power(2, env.N_NODES)+1)))[2:].zfill(env.N_NODES)
        return stateStr
    
    def computeStateList(self, x):
        if type(x) == str:
            stateList=[]
            for e in range(len(x)):
                stateList.append(int(x[e]))
        else:
            stateStr = self.computeStateStr(x)
            stateList=[]
            for e in range(len(stateStr)):
                stateList.append(int(stateStr[e]))
        return stateList
    
    def computeActionNbr(self, x):
        if type(x) == str:
            actionNbr = np.abs(np.power(2, env.N_INPUTS)-1-int(x,2))
        else:
            actionStr = ''.join(str(s) for s in x)
            actionNbr = np.abs(np.power(2, env.N_INPUTS)-1-int(actionStr,2))
        return actionNbr
    
    def computeActionStr(self, x):
        if type(x) == list:
            actionStr = ''.join(str(s) for s in x)
        else:
            actionStr = bin(np.abs((x-np.power(2, env.N_INPUTS)+1)))[2:].zfill(env.N_INPUTS)
        return actionStr
    
    def computeActionList(self, x):
        if type(x) == str:
            actionList=[]
            for e in range(len(x)):
                actionList.append(int(x[e]))
        else:
            actionStr = self.computeActionStr(x)
            actionList=[]
            for e in range(len(actionStr)):
                actionList.append(int(actionStr[e]))
        return actionList
    
    def step(self, ig, iu):
        prev_state = self.state
        rchoice = np.random.uniform()
        # here there are the dynamics of the network
        if rchoice>= 0.5:
            self.state = [int(ig[5] and ig[12]),
                          int(ig[24]), 
                          int(ig[1]), 
                          int(ig[27]), 
                          int(ig[20]), 
                          int(ig[4]),
                          int((ig[14] and iu[1]) or (ig[25] and iu[1])),
                          int(ig[13]), int(ig[17]),
                          int((ig[24] and ig[27])),
                          int(not ig[8]), int(ig[23]), int(ig[11]), int(ig[27]),
                          int(not ig[19] and iu[0] and iu[1]), int(ig[2]), int(not ig[10]), int(ig[1]),
                          int((ig[9] and ig[10] and ig[24] and ig[27]) or (ig[22] and ig[10] and ig[24] and ig[27])),
                          int((ig[6] or not ig[25])),
                          int((ig[10] or ig[21])),
                          int((ig[1] and ig[17])),
                          int(ig[14]), int(ig[17]), int(ig[7]),
                          int(not ig[3] and iu[2]),
                          int(ig[6] or (ig[14] and ig[25])),
                          int((not ig[3]) and ig[14] and ig[26])]
        else:
            self.state = [int(ig[5] and ig[12]),
                          int(ig[24]), 
                          int(ig[1]), 
                          int(ig[27]), 
                          int(ig[20]), 
                          int(ig[4]),
                          int((ig[14] and iu[1]) or (ig[25] and iu[1])),
                          int(ig[13]), int(ig[17]),
                          int((ig[24] and ig[27])),
                          int(not ig[8]), int(ig[23]), int(ig[11]), int(ig[27]),
                          int(not ig[19] and iu[0] and iu[1]), int(ig[2]), int(not ig[10]), int(ig[1]),
                          int((ig[9] and ig[10] and ig[24] and ig[27]) or (ig[22] and ig[10] and ig[24] and ig[27])),
                          int((ig[6] or not ig[25])),
                          int((ig[10] or ig[21])),
                          int((ig[1] and ig[17])),
                          int(ig[14]), int(ig[17]), int(ig[7]),
                          int(ig[25]),
                          int(ig[6] or (ig[14] and ig[25])),
                          int((not ig[3]) and ig[14] and ig[26])]
            
        # if DES_ATTR
        if self.state == env.DES_ATTR:
            self.cost = -1
            self.done = True
        # if there is a self loop but not DES_ATTR
        elif ((prev_state == self.state) and (self.state != env.DES_ATTR)):
            self.cost = 1
            self.done = False
        #non-significant state
        else:
            self.cost = 0
            self.done = False
        return self.state, self.cost, self.done
    

# %%
env = PBN_env()

# Agent properties
REPLAY_MEMORY_SIZE = int(1e6)  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE =  256 # How many samples to use for training
UPDATE_TARGET_MODEL_EVERY = 500
TAU = 0.005
HIDDEN_LAYER_1_LEN = 8
HIDDEN_LAYER_2_LEN = 8
HIDDEN_LAYER_3_LEN = 16
HIDDEN_LAYER_4_LEN = 8
LEARNING_RATE = 0.03

EPISODES = int(2e6)
SHOW_EVERY = 10000
DISCOUNT = 0.9
epsilons =  []
#epsilon greedy policy definition
START_EPSILON = 1
END_EPSILON = 0.01
END_POINT_EPSILON = int(EPISODES * 0.8)
epsilons = END_EPSILON*np.ones((EPISODES,1))
epsilons[0:END_POINT_EPSILON] = np.reshape(np.linspace(START_EPSILON, END_EPSILON,num= END_POINT_EPSILON), (END_POINT_EPSILON,1))

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


# %%
#########################
#####    AGENT    #######
#########################
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.model.set_weights(0*self.model.get_weights())
        self.target_model.set_weights(self.model.get_weights())
        
        self.Soft_Update = True # parameter update technique
        self.USE_PER = True
        
        if self.USE_PER:
            self.memory = Memory(REPLAY_MEMORY_SIZE)
        else:
            self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        
        model.add(Dense(HIDDEN_LAYER_1_LEN, activation="relu",input_dim=1))
        model.add(Dense(HIDDEN_LAYER_2_LEN, activation="relu",input_dim=HIDDEN_LAYER_1_LEN))
        model.add(Dense(HIDDEN_LAYER_3_LEN, activation="relu",input_dim=HIDDEN_LAYER_2_LEN))
        model.add(Dense(HIDDEN_LAYER_4_LEN, activation="relu",input_dim=HIDDEN_LAYER_3_LEN))
        model.add(Dense(env.action_space(), activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (s_t, a_t, g_(t+1), s_(t+1), done)
    def update_replay_memory(self, example):
        if self.USE_PER:
            self.memory.add(example)
        else:
            self.replay_memory.append(example)

    # Train main network during episode
    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved (MIN_REPLAY_MEMORY_SIZE)
        if self.USE_PER and self.memory.tree.n_entries < MIN_REPLAY_MEMORY_SIZE:
            return
        elif not self.USE_PER and len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch from replay memory
        if self.USE_PER:
            minibatch, idxs, is_weights = self.memory.sample(MINIBATCH_SIZE)
        else:
            minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then predict Q_t values
        current_states = np.array([example[0] for example in minibatch])
        current_q_values = self.model.predict(current_states)

        # Get future states from minibatch, then predict Q_(t+1) values
        new_current_states = np.array([example[3] for example in minibatch])
        future_q_values = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        actions_store = []
        absolute_errors = []
        for index, (current_state, action, cost, new_current_state, done) in enumerate(minibatch):
            actions_store.append(action)
            #take the index-th q values from the batches
            current_q_values_row = current_q_values[index]
            if not done:
                current_q_values_row[action] = cost + (DISCOUNT * np.min(future_q_values[index]))
            else:
                current_q_values_row[action] = cost

            # append to our training set
            X.append(current_state)
            y.append(current_q_values_row)
            absolute_errors.append(np.abs(current_q_values[index, action] - current_q_values_row[action]))
        
        if self.USE_PER:
            # update priority
            for i in range(len(minibatch)):
                idx = idxs[i]
                self.memory.update(idx, absolute_errors[i])
            
        # Train
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, epochs = 1, verbose=0, shuffle=True,sample_weight = is_weights)

        
        if self.Soft_Update:
            # Update target network every step
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-TAU) + q_weight * TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)
        else:
            # Update target network every UPDATE_TARGET_MODEL_EVERY episodes
            if terminal_state:
                self.target_update_counter += 1
            # If counter reaches set value, update target network with weights of main network
            if self.target_update_counter > UPDATE_TARGET_MODEL_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

    # Get q values from NN, given a state
    def get_q_values(self, state):
        c_state = state.reshape(-1, *state.shape) #just a reshaping from (n,) to (1,n)
        return self.model.predict(c_state)
    
    # Get q values from target NN, given a state
    def get_target_q_values(self, state):
        c_state = state.reshape(-1, *state.shape) #just a reshaping from (n,) to (1,n)
        return self.target_model.predict(c_state)

# %%
#########################
###  TRAINING PHASE  ####
#########################

#desired state (suppose that is known)
des = ['0000111000100000000110000110']
cost_avg = {des[0]: np.zeros((1, EPISODES // SHOW_EVERY))}
average_error= {des[0]: []}

T_END = 50 #steps in the episode
agent = {des[0]: DQNAgent()}

def live_plotter(x_vec,y1_data,line1,identifier='',pause_time=0.1):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)
        #update plot label/title
        plt.ylabel('Y Label')
        plt.ylim([0,1])
        plt.xlim([0, EPISODES])
        plt.title('Title: {}'.format(identifier))
        plt.show()
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_xdata(x_vec)
    line1.set_ydata(y1_data)
    plt.pause(pause_time)
    # return line so we can update it again in the next iteration
    return line1


for p in range(len(des)):
    costs = []
    env.DES_ATTR_string = des[p]
    env.DES_ATTR_nbr = env.computeStateNbr(env.DES_ATTR_string)
    env.DES_ATTR = env.computeStateList(env.DES_ATTR_string)
    print('TRAINING ON ' + env.DES_ATTR_string)
    line1 = []
    x_vec = []
    y_vec = []
    print('EPISODES NBR: ' + str(EPISODES))
    print(agent[des[p]].model.summary())
            
    episode = 0
    while episode < EPISODES:
        env.state = env.reset()
        curr_state = env.computeStateNbr(env.state)
        done = False
        
        for t in range(T_END-1):
            # epsilon greedy policy
            if np.random.random() > epsilons[episode]:
                # EXPLOITATION
                nbr_equal =np.where(agent[des[p]].get_q_values(np.array(curr_state)) == np.min(agent[des[p]].get_q_values(np.array(curr_state))))[1]
                if len(nbr_equal) == 1:
                    action = np.argmin(agent[des[p]].get_q_values(np.array(curr_state)))
                else:
                    action = random.choice(nbr_equal)
            else:
                # EXPLORATION
                action = np.random.randint(0, env.action_space())
            
            new_state, cost, done = env.step(env.computeStateList(curr_state), env.computeActionList(action))
            costs.append(cost)
            
            # Every step we update replay memory and train main network
            agent[des[p]].update_replay_memory((np.array(curr_state), action, cost,np.array(env.computeStateNbr(new_state)), done))
            agent[des[p]].train(done)
            curr_state = env.computeStateNbr(new_state)
            if done:
                break
            
        if  episode % SHOW_EVERY ==0:
            print('Episode: ' + str(episode))
            y_vec.append(np.mean(costs))
            x_vec.append(episode)
            line1 = live_plotter(x_vec,y_vec,line1)
            cost_avg[des[p]][0, episode // SHOW_EVERY] = np.mean(costs)
            costs = []
        episode += 1

# %%
#########################
####    PLOTS    ######
#########################

#plot AVG COST
plt.figure(2)
plt.title("Average Cost every " + str(SHOW_EVERY) + " episodes")
plt.plot(np.average(cost_avg[des[0]], axis=0))
plt.xlabel("Training Episodes/" + str(SHOW_EVERY))
plt.ylabel("Average Cost on " + des[0])

#GENE TRENDS
num_states = 1e4
num_actions = 40

x = np.zeros((num_states, num_actions, env.N_NODES))
for j in range(num_states):
    current_state = env.reset()
    env.state = current_state
    x[j,0, :] = env.state
    done = False
    for k in range(num_actions-1):
        action = np.argmin(agent[des[0]].get_target_q_values(np.array(env.computeStateNbr(current_state))))
        current_state, cost, done = env.step(current_state, env.computeActionList(action))
        x[j,k+1,:] = current_state

genesTrend = np.mean(x, axis = 0).T


options_colours = ('b', 'r')
colours = [options_colours[np.array(genesTrend[:,-1]>0.5, dtype=int)[c]] for c in range(np.size(genesTrend,0))]
labels = ['','']
labels_idx = [0,0]
done = False
count = 0
for i in range(env.N_NODES):
    if colours[i] == options_colours[0]:
        labels[0]= labels[0]+ 'X'+str(i)+', '
        labels_idx[0] = i
    else:
        labels[1]= labels[1]+ 'X'+str(i)+', '
        labels_idx[1] = i
labels[0] = labels[0][:-2]
labels[1] = labels[1][:-2]

plt.figure(3)
for i in range(env.N_NODES):
    if i == labels_idx[0]:
        plt.plot(np.linspace(0,num_actions-1,num_actions),genesTrend[i,:], colours[i], label = labels[0])
    elif i == labels_idx[1]:
        plt.plot(np.linspace(0,num_actions-1,num_actions),genesTrend[i,:], colours[i], label = labels[1])
    else:
        plt.plot(np.linspace(0,num_actions-1,num_actions),genesTrend[i,:], colours[i])
plt.legend()
plt.xlabel('Nbr od actions')
plt.ylabel('Average values')
plt.title('Average value of single genes over ' + str(num_states)+ ' initial states')
plt.xlim(0,num_actions)
plt.ylim(0, 1)
# %% save the model to disk
filename = 'finalized_model.sav'
pickle.dump(agent, open(filename, 'wb'))
pickle.dump(cost_avg, open('costs', 'wb'))
