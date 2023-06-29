import gym
import numpy as np
import torch as tr
from environment_utils import force, Box

class BoxEnvironment1(gym.Env):
    def __init__(self, space, goal):
        self.space = space
        self.goal = goal
    
    def init_state(self, agent_batch_size):
        self.state = np.zeros((agent_batch_size,5))
        self.state[:,0] = -0.5*np.ones(agent_batch_size)

    def step(self, action, U0, dt):
        x, y = self.state[:,0], self.state[:,1]
        theta = action[:,0]
        
        F_x, F_y = force(x,y,U0)

        e_x = np.cos(theta)
        v_x = e_x + F_x
        x_new = x + v_x*dt

        e_y = np.sin(theta)
        v_y = e_y + F_y
        y_new = y + v_y*dt
        inside_space = self.space.contains(np.array([x_new, y_new]).T)

        self.state[:,0][inside_space] = x_new[inside_space]
        self.state[:,1][inside_space] = y_new[inside_space]
        self.state[:,2] = F_x
        self.state[:,3] = F_y
        self.state[:,4] = theta

    def reward(self, dt):
        # Compute reward
        reward = -dt*np.ones(self.state.shape[0])/100
        wincondition = int(self.goal_check())
        reward += wincondition*100
        return reward
    
    def goal_check(self):
        position = self.state[:, 0:2]
        wincondition = self.goal.contains(position)
        return wincondition

