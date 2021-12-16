"""
Based on this paper: Evolution Strategies as a Scalable Alternative to Reinforcement Learning
Details can be found in : https://arxiv.org/abs/1703.03864

Exploration Enhangced ES
Give a bonus to the policy if the agent visits a 'different' state, and a penelty if a 'similar' state
The 'difference' is measured by the average distance between the current end state and nearest k end states

The model works well for some 'easy' mazes but gets stuck in local minimal when the maze is hard
"""
import numpy as np
#import gym
import multiprocessing as mp
from collections import deque
import time
import random

import matplotlib
import matplotlib.pyplot as plt

N_KID = 150                  # half of the training population
N_GENERATION = 1000         # training step
LR = .1                    # learning rate
ALPHA = 3
SIGMA = 1.2                # mutation strength or step size
N_CORE = mp.cpu_count()-1
maxlength = 10000    # maxlength of buffer
CONFIG = {'game':"CartPole-v0", 'n_feature':2, 'n_action':4, 'continuous_a':[False], 'ep_max_step':50}
   # choose your game

#maze = np.array([
#    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
#    [ 1.,  1.,  1.,  0.,  0.,  1.,  0.],
#    [ 0.,  0.,  0.,  1.,  1.,  1.,  0.],
#    [ 1.,  1.,  1.,  1.,  0.,  0.,  1.],
#    [ 1.,  0.,  0.,  0.,  1.,  1.,  1.],
#    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
#    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.]
#    ])
    
maze = np.array([
     [ 1.,  0.,  1.,  1.,  1.,  0.,  1.],
     [ 1.,  1.,  1.,  1.,  0.,  1.,  1.],
     [ 1.,  1.,  1.,  0.,  1.,  1.,  1.],
     [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
     [ 1.,  1.,  0.,  0.,  1.,  1.,  1.],
     [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
     [ 1.,  1.,  1.,  1.,  0.,  1.,  1.]
     ])
    
    
visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
rat_mark = 0.5      # The current rat cell will be painteg by gray 0.5
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)
rat_cell = (0,0)
obssize = 49
actsize = num_actions

###############################################################################
class Qmaze(object):
    def __init__(self, maze, rat=(0,0)):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (nrows-1, ncols-1)   # target cell where the "cheese" is
        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.0]
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not rat in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.reset(rat)

    def reset(self, rat):
        self.rat = rat
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = rat
        self.maze[row, col] = rat_mark
        self.state = (row, col, 'start')
        self.min_reward = 2* self.maze.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        valid_actions = self.valid_actions()
                
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:                  # invalid action, no change in rat position
            mode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows-1 and rat_col == ncols-1:
            return 100
        return 0
#        if mode == 'blocked':
#            return 0    #self.min_reward - 1
#        if (rat_row, rat_col) in self.visited:
#            return 0
#        if mode == 'invalid':
#            return 0
#        if mode == 'valid':
#            return 0

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status
    

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = rat_mark
        return canvas

    def game_status(self):
#         if self.total_reward < self.min_reward:
#             return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows-1 and rat_col == ncols-1:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows-1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols-1:
            actions.remove(2)

        if row>0 and self.maze[row-1,col] == 0.0:
            actions.remove(1)
        if row<nrows-1 and self.maze[row+1,col] == 0.0:
            actions.remove(3)

        if col>0 and self.maze[row,col-1] == 0.0:
            actions.remove(0)
        if col<ncols-1 and self.maze[row,col+1] == 0.0:
            actions.remove(2)

        return actions

# Implement replay buffer
class ReplayBuffer(object):
    
    def __init__(self, maxlength):
        """
        maxlength: max number of tuples to store in the buffer
        if there are more tuples than maxlength, pop out the oldest tuples
        """
        self.buffer = deque()
        self.number = 0
        self.maxlength = maxlength
    
    def append(self, experience):
        """
        this function implements appending new experience tuple
        experience: a tuple of the form (s,a,r,s^\prime)
        """
        self.buffer.append(experience)
        self.number += 1
        
    def pop(self):
        """
        pop out the oldest tuples if self.number > self.maxlength
        """
        while self.number > self.maxlength:
            self.buffer.popleft()
            self.number -= 1
    
    def sample(self, batchsize):
        """
        this function samples 'batchsize' experience tuples
        batchsize: size of the minibatch to be sampled
        return: a list of tuples of form (s,a,r,s^\prime)
        """
        # YOUR CODE HERE
        if batchsize < self.number:
            minibatch = random.sample(self.buffer, batchsize) 
        else:
            minibatch = random.sample(self.buffer, self.number) 
        return minibatch  # need implementation

def show(qmaze, file_name = 'Maze'):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6
    rat_row, rat_col, _ = qmaze.state
    canvas[rat_row, rat_col] = 0.3   # rat cell
    canvas[nrows-1, ncols-1] = 0.9 # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    plt.imshow(canvas, interpolation='none', cmap='gray')
    plt.savefig(file_name)
    return img
###############################################################################
        
def sign(k_id): return -1. if k_id % 2 == 0 else 1.  # mirrored sampling
#def process_obs(obs, maze_size = 7):
#    obs = obs.reshape(maze_size, maze_size)
#    for i in range(maze_size):
#        for j in range(maze_size):
#            if obs[i,j] == .5:
#                s = np.array([i, j])
#    def one_hot(a, num_classes):
#        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
#    obs_pro = one_hot(s, 7)
#    obs_pro = np.concatenate(obs_pro)
#    for i in range(row-1, row+2):
#        for j in range(col-1, col+2):
#            if i==-1 or i==7 or j==-1 or j==7:
#                s = 1.
#            else:
#                s = obs[i,j]
#            obs_pro.append(s)
#    return obs_pro
def process_obs(obs, maze_size = 7):
    obs = obs.reshape(maze_size, maze_size)
    for i in range(maze_size):
        for j in range(maze_size):
            if obs[i,j] == .5:
                return [i, j]


class SGD(object):                      # optimizer with momentum
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = np.zeros_like(params).astype(np.float32)
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v


def params_reshape(shapes, params):     # reshape to be a matrix
    p, start = [], 0
    for i, shape in enumerate(shapes):  # flat params to matrix
        n_w, n_b = shape[0] * shape[1], shape[1]
        p = p + [params[start: start + n_w].reshape(shape),
                 params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
        start += n_w + n_b
    return p

def get_curiosity(end_state, buffer, k):
    end_state = np.array(end_state)
    n = len(buffer)
    distances = np.zeros(n)
    for i in range(n):
        elem = np.array(buffer[i])
        distances[i] = np.sqrt(np.sum(np.square(end_state - elem)))
        
    top_k_indices = np.argsort(distances)[:min(k,n)]
    top_k = distances[top_k_indices]
    
    if (top_k==0).all():
        return -1
    return top_k.mean()


def get_end_state(shapes, params, env, ep_max_step, continuous_a, seed_and_id=None,):
    # perturb parameters using seed
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        params += sign(k_id) * SIGMA * np.random.randn(params.size)
    p = params_reshape(shapes, params)
    # run episode
    env.reset(rat_cell)
    s = env.observe()
    ep_r = 0.
    for step in range(ep_max_step):
        a = get_action(p, s, continuous_a)
        s, r, done = env.act(a)
        # mountain car's reward can be tricky
#        if env.spec._env_name == 'MountainCar' and s[0] > -0.1: r = 0.
        ep_r += r
        if done != 'not_over': break
    
    return process_obs(s[0])


def get_action(params, x, continuous_a):
    x = np.array( [process_obs(elem) for elem in x])
    x = np.tanh(x.dot(params[0]) + params[1])
    x = x.dot(params[2]) + params[3]
    if not continuous_a[0]: return np.argmax(x, axis=1)[0]      # for discrete action
    else: return continuous_a[1] * np.tanh(x)[0]                # for continuous action


def build_net():
    def linear(n_in, n_out):  # network linear layer
        w = np.random.randn(n_in * n_out).astype(np.float32) * .1
        b = np.random.randn(n_out).astype(np.float32) * .1
        return (n_in, n_out), np.concatenate((w, b))
    s0, p0 = linear(CONFIG['n_feature'], 8)
    s1, p1 = linear(8, CONFIG['n_action'])
    return [s0, s1], np.concatenate((p0,p1))


#def train(net_shapes, net_params, buffer, k, optimizer, utility, pool):
#    # pass seed instead whole noise matrix to parallel will save your time
#    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)    # mirrored sampling
#
#    # distribute training in parallel
#    jobs = [pool.apply_async(get_end_state, (net_shapes, net_params, env, CONFIG['ep_max_step'], CONFIG['continuous_a'],
#                                          [noise_seed[k_id], k_id], )) for k_id in range(N_KID*2)]
#
#    end_states = [j.get() for j in jobs]
#    rewards = np.array([get_curiosity(end_state, buffer, k) for end_state in end_states])  
#    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward
#
#    cumulative_update = np.zeros_like(net_params)       # initialize update values
#    for ui, k_id in enumerate(kids_rank):
#        np.random.seed(noise_seed[k_id])                # reconstruct noise using seed
#        cumulative_update += utility[ui] * sign(k_id) * np.random.randn(net_params.size)
#
#    gradients = optimizer.get_gradients(cumulative_update/(2*N_KID*SIGMA))
#    
#    add_to_buffer = random.sample(end_states, N_KID)
#    
#    return net_params + gradients, rewards, add_to_buffer

def train(net_shapes, net_params, buffer, k, pool):
    # pass seed instead whole noise matrix to parallel will save your time
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)    # mirrored sampling

    # distribute training in parallel
    jobs = [pool.apply_async(get_end_state, (net_shapes, net_params, env, CONFIG['ep_max_step'], CONFIG['continuous_a'],
                                          [noise_seed[k_id], k_id], )) for k_id in range(N_KID*2)]

    end_states = [j.get() for j in jobs]
    rewards = np.array([get_curiosity(end_state, buffer, k) for end_state in end_states]) 
#    A = (rewards - np.mean(rewards)) / (np.std(rewards)+1e-5)
    
    
    G = np.zeros((2*N_KID, net_params.size))
    for k_id in range(2*N_KID):
        np.random.seed(noise_seed[k_id]) 
        G[k_id] = sign(k_id) * np.random.randn(net_params.size)
        
    gradients = ALPHA / (2 * N_KID * SIGMA) * np.dot(G.T, rewards)
    
    add_to_buffer = random.sample(end_states, 10)
    
    return net_params + gradients, rewards, add_to_buffer


if __name__ == "__main__":
    # utility instead reward for update parameters (rank transformation)
    base = N_KID * 2    # *2 for mirrored sampling
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base
    k = 20
    
    # training
    net_shapes, net_params = build_net()
#    env = gym.make(CONFIG['game']).unwrapped
    buffer = ReplayBuffer(maxlength)
    env = Qmaze(maze)
    optimizer = SGD(net_params, LR)
    pool = mp.Pool(processes=N_CORE)
    mar = None      # moving average reward
    
    end_state = get_end_state(net_shapes, net_params, env, CONFIG['ep_max_step'], CONFIG['continuous_a'], None,)
    buffer.append(end_state)
    
    net_ends = []
    for g in range(N_GENERATION):
        t0 = time.time()
#        net_params, kid_rewards, add_to_buffer = train(net_shapes, net_params, buffer.buffer, k, optimizer, utility, pool)
        net_params, kid_rewards, add_to_buffer = train(net_shapes, net_params, buffer.buffer, k, pool)
        # test trained net without noise      
        net_end =  get_end_state(net_shapes, net_params, env, CONFIG['ep_max_step'], CONFIG['continuous_a'], None,)
        net_ends.append(net_end)
        net_r = get_curiosity(net_end, buffer.buffer, k)
        mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
        
        # Update the buffer
        for elem in add_to_buffer:
            buffer.append(elem)
            buffer.pop()
        buffer.append(net_end)
        buffer.pop()
        
        #Output test maze graph
        # run episodes
        env.reset(rat_cell)
        done = 'not_over'
        s = env.observe()
        for step in range(CONFIG['ep_max_step']):
            a = get_action(net_params, s, CONFIG['continuous_a'])
            s, r, done = env.act(a)
            if done != 'not_over': break
        show(env, file_name = 'es_Maze'+'test_1_'+str(g+1))
        
#        SIGMA = max(.1, SIGMA * .995)
#        ALPHA = max(.1, ALPHA * .999)
        
        print(
            'Gen: ', g,
            '| Net_R: %.1f' % net_r,
            '| Kid_avg_R: %.1f' % kid_rewards.mean(),
            '| Gen_T: %.2f' % (time.time() - t0),
            '| Net_end:{}'.format(net_end),)
#
#    # test
#    print("\nTESTING....")
#    p = params_reshape(net_shapes, net_params)
#    while True:
#        s = env.reset()
#        for _ in range(CONFIG['ep_max_step']):
#            env.render()
#            a = get_action(p, s, CONFIG['continuous_a'])
#            s, _, done, _ = env.step(a)
#            if done: break