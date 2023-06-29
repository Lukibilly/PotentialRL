import matplotlib.pyplot as plt
import numpy as np

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
