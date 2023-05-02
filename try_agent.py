#%%

import pickle
import torch
from time import sleep

from hard_maze import Hard_Maze

which_args = "en_hard"
which_epoch = 1000
which_agent = 1
maze = "t"

def try_agent(which_args, which_epoch, which_agent, maze):
    with open("saved/{}/plot_dict.pickle".format(which_args), "rb") as handle: 
        plot_dict = pickle.load(handle)
    args = plot_dict["args"]
    agent_lists = plot_dict["agent_lists"]
    state_dict = agent_lists["{}_{}".format(which_agent, which_epoch)]
    actor = agent_lists["actor"](args)
    actor.load_state_dict(state_dict[1])
    
    maze = Hard_Maze(maze, True, args)
    sleep(2)
    done = False
    prev_a = torch.zeros((1, 1, 2))
    h_actor = torch.zeros((1, 1, args.hidden_size))
    while(done == False):
        sleep(1)
        o, s = maze.obs()
        a, _, h_actor = actor(o, s, prev_a, h_actor)
        action = torch.flatten(a).tolist()
        _, _, done, _ = maze.action(action[0], action[1], True)
        prev_a = a
    maze.maze.stop()
    
try_agent(which_args, which_epoch, which_agent, maze)
# %%
