#%%

import torch
import enlighten

from utils import device, args, T_Maze, plot_rewards, plot_spot_names
from agent import Agent

def episode(agent, push = True, verbose = False):
    done = False
    t_maze = T_Maze()
    steps = 0
    if(verbose): print("\nSTART!\n")
    with torch.no_grad():
        while(done == False):
            if(verbose): print(t_maze)
            o = t_maze.agent_pos
            a = agent.act(torch.tensor(o).unsqueeze(0).float())
            r, spot_name = t_maze.action(a[0], a[1])
            no = t_maze.agent_pos
            steps += 1
            if(r != 0): done = True
            if(steps >= agent.args.max_steps): done = True ; r = -1
            if(device == "cuda"): torch.cuda.synchronize(device=device)
            if(push): agent.memory.push(o, a, r, no, done, done, agent)
    if(verbose): print(t_maze)
    return(r, spot_name)



class Trainer():
    def __init__(self, args = args):
        
        self.args = args
        self.restart()
    
    def restart(self):
        self.e = 0
        self.agent = Agent(args = self.args)

    def train(self):
        self.agent.train()
        manager = enlighten.Manager()
        E = manager.counter(total = self.args.epochs, desc = "Epochs:", unit = "ticks", color = "blue")
        rewards = [] 
        spot_names = []
        while(True):
            E.update()
            self.e += 1
            r, spot_name = episode(self.agent)
            rewards.append(r) ; spot_names.append(spot_name)
            losses, extrinsic, intrinsic_curiosity, intrinsic_entropy, dkl_change = \
                self.agent.learn(batch_size = self.args.batch_size)
            if(self.e % 100 == 0):
                plot_rewards(rewards, self.e)
                plot_spot_names(spot_names, self.e)
                episode(self.agent, push = False, verbose = True)
            if(self.e >= self.args.epochs): 
                break
    
print("train.py loaded.")
# %%