import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

class RLPlotter():
    def __init__(self, logger):
        self.logger = logger
        self.plot_episode_actions()
        self.plot_episode_losses()
        self.plot_episode_steps()

    def plot_episode_losses(self):
        folder = os.path.join('logs', 'episode_losses')
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        for i in range(len(self.logger.episode_losses)):
            x = np.arange(len(self.logger.episode_losses[i]))
            plt.plot(x, self.logger.episode_losses[i])
            plt.xlabel('step')
            plt.ylabel('loss')
            plt.savefig(os.path.join(folder, f'episode{i}_losses.png'))
            plt.clf()

    def plot_episode_steps(self):
        folder = os.path.join('logs', 'episode_steps')
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        x = np.arange(len(self.logger.episode_steps))
        plt.plot(x, self.logger.episode_steps)
        plt.xlabel('episode')
        plt.ylabel('steps')
        plt.savefig(os.path.join(folder, 'episode_steps.png'))
        plt.clf()

    def plot_episode_actions(self):
        folder = os.path.join('logs', 'episode_actions')
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        for i in range(len(self.logger.episode_actions)):
            x = np.arange(len(self.logger.episode_actions[i]))
            plt.plot(x, self.logger.episode_actions[i])
            plt.xlabel('step')
            plt.ylabel('action')
            plt.savefig(os.path.join(folder, f'episode{i}_actions.png'))
            plt.clf()


def plot_normalized_mexican_hat_potential():
    x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    bool = (x**2+y**2)**0.5>0.5
    Potential = 16*(x**2+y**2-0.25)**2
    Potential[bool] = 0
    plt.xlim(-0.8,0.8)
    plt.ylim(-0.8,0.8)
    plt.scatter(-0.5,0,c='black',label='start',marker='D')    
    plt.scatter(0.5,0,c='black',label='goal',marker='x')

    plt.imshow(Potential,cmap = 'Greys',extent=[-1,1,-1,1],origin='lower')
    colorbar = plt.colorbar()
    colorbar.set_label(r'$U/U_0$',labelpad=10,fontsize = 20)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

def plot_losses(logger):
    plt.plot(logger.losses)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('Loss over time')
    plt.show()
