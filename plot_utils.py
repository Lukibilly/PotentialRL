import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

class RLPlotter():
    def __init__(self, logger):
        self.logger = logger

    def clear_plots(self):
        clear_folder('episode_losses')
        clear_folder('episode_actions')
        clear_folder('episode_paths')
        clear_folder('run_losses')
        clear_folder('episode_steps')

    def plot_last_episode(self):
        self.plot_last_episode_losses()
        self.plot_last_episode_actions()
        self.plot_last_episode_paths()
        self.plot_last_losses()
        self.plot_last_episode_steps()

    def plot_last_episode_losses(self):        
        folder = os.path.join('logs', 'episode_losses')
        i = len(self.logger.episode_losses)-1
        x = np.arange(len(self.logger.episode_losses[i]))
        plt.plot(x, self.logger.episode_losses[i])
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title(f'Episode {i} MSE')
        plt.savefig(os.path.join(folder, f'episode{i}_losses.png'))
        plt.clf()

    def plot_last_losses(self):
        folder = os.path.join('logs', 'run_losses')
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        x = np.arange(len(self.logger.losses))
        plt.plot(x, self.logger.losses)
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title('Run MSE')
        plt.savefig(os.path.join(folder, 'run_losses.png'))
        plt.clf()

    def plot_last_episode_steps(self):
        folder = os.path.join('logs', 'episode_steps')
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        x = np.arange(len(self.logger.episode_steps))
        plt.plot(x, self.logger.episode_steps)
        plt.xlabel('episode')
        plt.ylabel('steps')
        plt.savefig(os.path.join(folder, 'episode_steps.png'))
        plt.clf()
    
    def plot_last_episode_actions(self):
        folder = os.path.join('logs', 'episode_actions')
        i = len(self.logger.episode_actions) - 1
        x = np.arange(len(self.logger.episode_actions[i]))
        plt.plot(x, self.logger.episode_actions[i])
        plt.xlabel('step')
        plt.ylabel('action')
        plt.savefig(os.path.join(folder, f'episode{i}_actions.png'))
        plt.clf()

    def plot_last_episode_paths(self):
        folder = os.path.join('logs', 'episode_paths')
        i = len(self.logger.episode_states) - 1
        x = np.array(self.logger.episode_states[i])[:,0,0]
        y = np.array(self.logger.episode_states[i])[:,0,1]
        plot_normalized_mexican_hat_potential()
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Episode {i} Path')
        plt.savefig(os.path.join(folder, f'episode{i}_path.png'))
        plt.clf()

def plot_normalized_mexican_hat_potential():
    x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    bool = (x**2+y**2)**0.5>0.5
    Potential = 16*(x**2+y**2-0.25)**2
    Potential[bool] = 0
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.scatter(-0.5,0,c='black',label='start',marker='D')    
    plt.scatter(0.5,0,c='black',label='goal',marker='x')

    plt.imshow(Potential,cmap = 'Greys',extent=[-1,1,-1,1],origin='lower')
    colorbar = plt.colorbar()
    colorbar.set_label(r'$U/U_0$',labelpad=10,fontsize = 20)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

def clear_folder(folder_name):
    folder = os.path.join('logs', folder_name)
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)


