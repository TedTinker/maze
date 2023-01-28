#%%

import torch
import enlighten
from itertools import accumulate
from copy import deepcopy

from utils import default_args
from maze import T_Maze
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
            r, spot_name, done = t_maze.action(a.tolist())
            no = t_maze.obs()
            steps += 1
            if(steps >= agent.args.max_steps): done = True ; r = -1
            if(push): agent.memory.push(o, a, r, no, done, done, agent)
    if(verbose): print(t_maze)
    return(r, spot_name)



class Trainer():
    def __init__(self, args = default_args, title = None):
        
        self.args = args
        self.title = title
        self.restart()
    
    def restart(self):
        self.e = 0
        self.agents = [Agent(args = self.args) for _ in range(self.args.agents)]
        self.plot_dict = {
            "title" : self.title,
            "rewards" : [[] for agent in self.agents], "spot_names" : [[] for agent in self.agents], 
            "mse" : [[] for agent in self.agents], "dkl" : [[] for agent in self.agents], 
            "alpha" : [[] for agent in self.agents], "actor" : [[] for agent in self.agents], 
            "critic_1" : [[] for agent in self.agents], "critic_2" : [[] for agent in self.agents], 
            "extrinsic" : [[] for agent in self.agents], "intrinsic_curiosity" : [[] for agent in self.agents], 
            "intrinsic_entropy" : [[] for agent in self.agents], "dkl_change" : [[] for agent in self.agents],
            "naive" : [[] for agent in self.agents], "friston" : [[] for agent in self.agents]}

    def train(self):
        for agent in self.agents: agent.train()
        manager = enlighten.Manager()
        E = manager.counter(total = self.args.epochs, desc = "{}:".format(self.title), unit = "ticks", color = "blue")
        while(True):
            E.update()
            self.e += 1
            for i, agent in enumerate(self.agents):
                r, spot_name = episode(agent)
                l, e, ic, ie, dkl, naive, friston = agent.learn(batch_size = self.args.batch_size)
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
                self.plot_dict["naive"][i].append(naive)
                self.plot_dict["friston"][i].append(friston)
            if(self.e >= self.args.epochs): 
                break
        for i, rewards in enumerate(self.plot_dict["rewards"]):
            self.plot_dict["rewards"][i] = list(accumulate(rewards))
        min_max_dict = {key : [] for key in self.plot_dict.keys()}
        for key in min_max_dict.keys():
            if(not key in ["title", "spot_names"]):
                minimum = None ; maximum = None 
                for l in self.plot_dict[key]:
                    l = deepcopy(l)
                    l = [_ for _ in l if _ != None]
                    if(l != []):
                        if(minimum == None):    minimum = min(l)
                        elif(minimum > min(l)): minimum = min(l)
                        if(maximum == None):    maximum = max(l) 
                        elif(maximum < max(l)): maximum = max(l)
                min_max_dict[key] = (minimum, maximum)
        return(self.plot_dict, min_max_dict)
    
print("train.py loaded.")
# %%