#%%

import torch
import enlighten
from copy import deepcopy

from utils import device, args, T_Maze, plots
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
    def __init__(self, args = args, title = None):
        
        self.args = args
        self.title = title
        self.restart()
    
    def restart(self):
        self.e = 0
        self.agents = [Agent(args = self.args) for _ in range(self.args.agents)]
        self.plot_dict = {
            "rewards" : [[] for agent in self.agents], "spot_names" : [[] for agent in self.agents], 
            "mse" : [[] for agent in self.agents], "dkl" : [[] for agent in self.agents], 
            "alpha" : [[] for agent in self.agents], "actor" : [[] for agent in self.agents], 
            "critic_1" : [[] for agent in self.agents], "critic_2" : [[] for agent in self.agents], 
            "extrinsic" : [[] for agent in self.agents], "intrinsic_curiosity" : [[] for agent in self.agents], 
            "intrinsic_entropy" : [[] for agent in self.agents], "dkl_change" : [[] for agent in self.agents]
        }

    def train(self):
        for agent in self.agents: agent.train()
        manager = enlighten.Manager()
        E = manager.counter(total = self.args.epochs, desc = "Epochs:", unit = "ticks", color = "blue")
        while(True):
            E.update()
            self.e += 1
            for i, agent in enumerate(self.agents):
                r, spot_name = episode(agent)
                l, e, ic, ie, dkl = agent.learn(batch_size = self.args.batch_size)
                self.plot_dict["rewards"][i].append(r)
                self.plot_dict["spot_names"][i].append(spot_name)
                self.plot_dict["mse"][i].append(l[0][0])
                self.plot_dict["dkl"][i].append(l[0][1])
                self.plot_dict["alpha"][i].append(l[0][2])
                self.plot_dict["actor"][i].append(l[0][3])
                self.plot_dict["critic_1"][i].append(l[0][4])
                self.plot_dict["critic_2"][i].append(l[0][5])
                self.plot_dict["extrinsic"][i].append(e)
                self.plot_dict["intrinsic_curiosity"][i].append(ic)
                self.plot_dict["intrinsic_entropy"][i].append(ie)
                self.plot_dict["dkl_change"][i].append(dkl)
            if(self.e % 100 == 0):
                plots(deepcopy(self.plot_dict), self.title)
                episode(self.agents[0], push = False, verbose = True)
            if(self.e >= self.args.epochs): 
                break
    
print("train.py loaded.")
# %%