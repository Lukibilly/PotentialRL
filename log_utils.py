import numpy as np
import matplotlib.pyplot as plt

class RLLogger():
    def __init__(self):
        self.states = []
        self.actions = []
        self.losses = []
        self.episode_states = []
        self.episode_actions = []
        self.episode_losses = []
        self.episode_steps = []

    def save_state(self, state):
        self.states.append(state)

    def save_episode(self, steps):
        self.episode_states.append(self.states[-steps:])
        self.episode_actions.append(self.actions[-steps:])
        self.episode_losses.append(self.losses[-steps:])
        self.episode_steps.append(steps)

    def save_loss(self, loss):
        self.losses.append(loss)
    
    def save_action(self, action):
        self.actions.append(action)

    

        