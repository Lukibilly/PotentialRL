import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

class RLPlotter():
    def __init__(self, logger, goal_area):
        self.logger = logger
        self.goal_area = goal_area

    def clear_plots(self):
        clear_folder('episode_losses_critic')
        clear_folder('episode_losses_actor')
        clear_folder('episode_actions')
        clear_folder('episode_paths')
        clear_folder('run_losses_critic')
        clear_folder('run_losses_actor')
        clear_folder('episode_steps')
        clear_folder('episode_actions_polar')

    def plot_last_episode(self):
        self.plot_last_episode_losses_critic()
        self.plot_last_episode_losses_actor()
        self.plot_last_episode_actions()
        self.plot_last_episode_paths()
        self.plot_last_losses_critic()
        self.plot_last_losses_actor()
        self.plot_last_episode_steps()
        self.plot_last_episdode_actions_polar()

    def plot_last_episode_losses_critic(self):        
        folder = os.path.join('logs', 'episode_losses_critic')
        i = len(self.logger.episode_losses_critic)-1
        x = np.arange(len(self.logger.episode_losses_critic[i]))
        plt.plot(x, self.logger.episode_losses_critic[i])
        plt.xlabel('Step')
        plt.ylabel('Loss Critic')
        plt.title(f'Episode {i} MSE')
        plt.savefig(os.path.join(folder, f'episode{i}_losses_critic.png'))
        plt.close()
    
    def plot_last_episode_losses_actor(self):
        folder = os.path.join('logs', 'episode_losses_actor')
        i = len(self.logger.episode_losses_actor)-1
        x = np.arange(len(self.logger.episode_losses_actor[i]))
        plt.plot(x, self.logger.episode_losses_actor[i])
        plt.xlabel('Step')
        plt.ylabel('Loss Actor')
        plt.title(f'Episode {i} MSE')
        plt.savefig(os.path.join(folder, f'episode{i}_losses_actor.png'))
        plt.close()

    def plot_last_losses_critic(self):
        folder = os.path.join('logs', 'run_losses_critic')
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        x = np.arange(len(self.logger.losses_critic))
        plt.plot(x, self.logger.losses_critic)
        plt.xlabel('Step')
        plt.ylabel('Loss Critic')
        plt.title('Run MSE')
        plt.savefig(os.path.join(folder, 'run_losses_critic.png'))
        plt.close()
    
    def plot_last_losses_actor(self):
        folder = os.path.join('logs', 'run_losses_actor')
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        x = np.arange(len(self.logger.losses_actor))
        plt.plot(x, self.logger.losses_actor)
        plt.xlabel('Step')
        plt.ylabel('Loss Actor')
        plt.title('Run MSE')
        plt.savefig(os.path.join(folder, 'run_losses_actor.png'))
        plt.close()

    def plot_last_episode_steps(self):
        folder = os.path.join('logs', 'episode_steps')
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        x = np.arange(len(self.logger.episode_steps))
        plt.plot(x, self.logger.episode_steps)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.savefig(os.path.join(folder, 'episode_steps.png'))
        plt.close()
    
    def plot_last_episode_actions(self):
        folder = os.path.join('logs', 'episode_actions')
        i = len(self.logger.episode_actions) - 1
        x = np.arange(len(self.logger.episode_actions[i]))
        plt.plot(x, self.logger.episode_actions[i])
        plt.xlabel('Step')
        plt.ylabel(r'Active Orientation $\theta$')
        plt.ylim(0,7)
        plt.savefig(os.path.join(folder, f'episode{i}_actions.png'))
        plt.close()
    
    def plot_last_episdode_actions_polar(self):
        folder = os.path.join('logs', 'episode_actions_polar')
        i = len(self.logger.episode_actions) - 1
        x = np.arange(len(self.logger.episode_actions[i]))
        fig = plt.figure()
        ax = plt.subplot(111, polar=True)
        ax.scatter(self.logger.episode_actions[i], x, c=x, cmap = 'plasma')
        fig.savefig(os.path.join(folder, f'episode{i}_actions_polar.png'))
        plt.close()

    def plot_last_episode_paths(self):
        folder = os.path.join('logs', 'episode_paths')
        i = len(self.logger.episode_states) - 1
        x = np.array(self.logger.episode_states[i])[:,0,0]
        y = np.array(self.logger.episode_states[i])[:,0,1]
        plot_normalized_mexican_hat_potential(self.goal_area)
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Episode {i} Path')
        plt.savefig(os.path.join(folder, f'episode{i}_path.png'))
        plt.close()

def plot_normalized_mexican_hat_potential(goal_area):
    x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    bool = (x**2+y**2)**0.5>0.5
    Potential = 16*(x**2+y**2-0.25)**2
    Potential[bool] = 0
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.scatter(-0.5,0,c='black',label='start',marker='D')    
    plt.scatter(0.5,0,c='black',label='goal',marker='x')
    plt.Rectangle((goal_area.centerX, goal_area.centerY), goal_area.width, goal_area.height, color='black', fill=False, alpha=0.5, label='goal area')
    plt.imshow(Potential,cmap = 'Greys',extent=[-1,1,-1,1],origin='lower')
    colorbar = plt.colorbar()
    colorbar.set_label(r'$U/U_0$',labelpad=10,fontsize = 20)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

def clear_folder(folder_name):
    folder = os.path.join('logs', folder_name)
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)


