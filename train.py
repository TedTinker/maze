#%%

import torch
import enlighten

from utils import device, args, T_Maze, plot_rewards, plot_spot_names, plot_losses, plot_ext_int, plot_dkl_change
from agent import Agent

def episode(agent, push = True, verbose = False):
    done = False
    t_maze = T_Maze()
    steps = 0
    if(verbose): print("\nSTART!\n")
    with torch.no_grad():
        while(done == False):
            if(verbose): print(t_maze)
            o = t_maze.obs()
            a = agent.act(o)
            r, spot_name, done = t_maze.action(a[0], a[1])
            no = t_maze.obs()
            steps += 1
            if(steps >= agent.args.max_steps): done = True ; r = -1
            if(device == "cuda"): torch.cuda.synchronize(device=device)
            if(push): agent.memory.push(o, a, r, no, done, done, agent)
    if(verbose): print(t_maze)
    return(r, spot_name)



class Trainer():
    def __init__(self, args = args, folder = None):
        
        self.args = args
        self.restart()
    
    def restart(self):
        self.e = 0
        self.agent = Agent(args = self.args)

    def train(self):
        self.agent.train()
        manager = enlighten.Manager()
        E = manager.counter(total = self.args.epochs, desc = "Epochs:", unit = "ticks", color = "blue")
        rewards = [] ; spot_names = []
        losses = [] ; extrinsic = [] ; intrinsic_curiosity = [] ; intrinsic_entropy = [] ; dkl_change = []
        while(True):
            E.update()
            self.e += 1
            r, spot_name = episode(self.agent)
            rewards.append(r) ; spot_names.append(spot_name)
            l, e, ic, ie, dkl = self.agent.learn(batch_size = self.args.batch_size)
            losses.append(l) ; extrinsic.append(e) ; intrinsic_curiosity.append(ic)
            intrinsic_entropy.append(ie) ; dkl_change.append(dkl)
            if(self.e % 100 == 0):
                plot_rewards(rewards, self.e)
                plot_spot_names(spot_names, self.e)
                plot_losses(losses, e)
                plot_ext_int(extrinsic, intrinsic_curiosity, intrinsic_entropy, self.e)
                plot_dkl_change(dkl_change, self.e)
                episode(self.agent, push = False, verbose = True)
            if(self.e >= self.args.epochs): 
                break
    
print("train.py loaded.")
# %%