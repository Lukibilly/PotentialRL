import gym
import numpy as np
import torch as tr

def force(x,y,U0,type='toy'):
    if type == 'mexican':
        r = tr.sqrt(x**2+y**2)
        bool = r>0.5
        fr = -64*U0*(r**2-0.25)
        fr[bool] = 0
        F_x = fr*x
        F_y = fr*y
        return F_x,F_y
    if type == 'toy':
        F_x = np.zeros(x.shape) #0.1*tr.sin(2*np.pi*x)
        F_y = 0.1*np.sin(x)
        return F_x,F_y
    
class Box():
    def __init__(self, width, height, center=None):
        self.width = width
        self.height = height
        if center is None:
            self.centerX = 0
            self.centerY = 0
        else:      
            self.centerX = center[:,0]
            self.centerY = center[:,1]
    
    def contains(self, state):
        x, y = state[:, 0], state[:, 1]
        bool = np.logical_and(np.logical_and(x > self.centerX-self.width/2, x < self.centerX+self.width/2),
                              np.logical_and(y > self.centerY-self.height/2, y < self.centerY+self.height/2))
        return bool