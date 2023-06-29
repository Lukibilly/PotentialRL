import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

class RLLogger():
    def __init__(self):
        self.states = []
        self.losses = []
        self.episode_states = []
        self.episode_losses = []
        self.episode_steps = []

    def save_state(self, state):
        self.states.append(state)

    def save_episode(self, steps):
        self.episode_states.append(self.states[-steps:])
        self.episode_losses.append(self.losses[-steps:])
        self.episode_steps.append(steps)

    def save_loss(self, loss):
        self.losses.append(loss)

    def plot_episode_losses(self):
        folder = os.path.join('logs', 'episode_losses')
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        for i in range(len(self.episode_losses)):
            x = np.arange(len(self.episode_losses[i]))
            plt.plot(x, self.episode_losses[i])
            plt.savefig(os.path.join(folder, f'episode{i}_losses.png'))
            plt.clf()

    def plot_episode_steps(self):
        folder = os.path.join('logs', 'episode_steps')
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        x = np.arange(len(self.episode_steps))
        plt.plot(x, self.episode_steps)
        plt.savefig(os.path.join(folder, 'episode_steps.png'))
        plt.clf()
    

        