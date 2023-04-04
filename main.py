#%%

import pickle, os, torch, random
import numpy as np
from multiprocessing import Pool

from utils import args, folder, duration
from agent import Agent

print("\nname:\n{}".format(args.arg_name))
print("\nagents: {}. previous_agents: {}.".format(args.agents, args.previous_agents))

def train(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    agent = Agent(args)
    plot_dict, min_max_dict = agent.training(str(i).zfill(3))
    with open(folder + "/plot_dict_{}.pickle".format(   str(i).zfill(3)), "wb") as handle:
        pickle.dump(plot_dict, handle)
    with open(folder + "/min_max_dict_{}.pickle".format(str(i).zfill(3)), "wb") as handle:
        pickle.dump(min_max_dict, handle)
        
with Pool() as p: 
    p.map(train, range(1 + args.previous_agents, args.agents + 1 + args.previous_agents))
    p.close() ; p.join()
        
print("Done with {}!".format(args.arg_name))
print("\n\nDuration: {}".format(duration()))
# %%
