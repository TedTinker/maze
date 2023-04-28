#%%
from random import choice, choices
import pandas as pd
import torch

from utils import default_args, args, print



class Spot:
    
    def __init__(self, pos, exit_reward = None, name = "NONE"):
        self.pos = pos ; self.exit_reward = exit_reward ; self.name = name
        
        
        
class Arena_Description:
    def __init__(self, start, spots, random_pos = []):
        self.start = start ; self.spots = spots ; self.random_pos = random_pos
        self.xs = list(set([spot.pos[0] for spot in spots])) ; self.xs.sort()
        self.ys = list(set([spot.pos[1] for spot in spots])) ; self.ys.sort() ; self.ys.reverse()
        print(self.xs, self.ys)
        
        
        
arena_dict = {
    "t" : Arena_Description(
        (0, 0), 
        [Spot((0, 0)), Spot((0, 1)), 
         Spot((-1, 1), args.default_reward, "BAD"), Spot((1, 1)), Spot((1, 2)), 
         Spot((2, 2)), Spot((3, 2)), Spot((3, 1), args.better_reward, "GOOD")],
        [(0, 1)])
}
            
        

class Easy_Maze:
    
    def __init__(self, maze_name, args = default_args):
        self.args = args
        self.maze = arena_dict[maze_name]
        self.begin()
        
    def begin(self):
        self.steps = 0 
        self.agent_pos = self.maze.start
        
    def obs(self):
        pos = [1 if spot.pos == self.agent_pos else self.args.non_one for spot in self.maze.spots]
        random_spot = False
        right = self.args.non_one ; left = self.args.non_one ; up = self.args.non_one ; down = self.args.non_one
        for spot in self.maze.spots:
            if(spot.pos == (self.agent_pos[0]+1, self.agent_pos[1])):   right = 1 
            if(spot.pos == (self.agent_pos[0]-1, self.agent_pos[1])):   left = 1  
            if(spot.pos == (self.agent_pos[0],   self.agent_pos[1]+1)): up = 1    
            if(spot.pos == (self.agent_pos[0],   self.agent_pos[1]-1)): down = 1  
            if(spot.pos == self.agent_pos): random_spot = spot.pos in self.maze.random_pos
        pos += [right, left, up, down]
        for _ in range(self.args.randomness): pos += [choice([-1,1]) if random_spot else 0]
        return(torch.tensor(pos).unsqueeze(0).float())
    
    def obs_str(self):
        obs = self.obs().squeeze(0)
        spot_num = torch.argmax(obs[:-4]).item()
        r = bool(obs[-4].item()==1)
        l = bool(obs[-3].item()==1)
        u = bool(obs[-2].item()==1)
        d = bool(obs[-1].item()==1)
        return("Observation: Spot #{}. Right {}. Left {}. Up {}. Down {}.".format(
            spot_num, r, l, u, d))
        
    def action(self, x, y, verbose = False):
        if(abs(x) > abs(y)): y = 0 ; x = 1 if x > 0 else -1
        else:                x = 0 ; y = 1 if y > 0 else -1 
        new_pos = (self.agent_pos[0] + x, self.agent_pos[1] + y)
        
        self.steps += 1
        wall = True ; exit = False ; reward = 0 ; spot_name = "NONE" ; done = False
        
        for spot in self.maze.spots:
            if(spot.pos == new_pos):
                wall = False
                self.agent_pos = new_pos ; reward = 0 ; spot_name = spot.name
                if(spot.exit_reward != None):
                    done = True ; exit = True
                    weights = [w for w, r in spot.exit_reward]
                    rewards = [r for w, r in spot.exit_reward]
                    reward = choices(rewards, weights = weights, k = 1)[0]
        
        if(wall): reward += self.args.wall_punishment
        if(self.steps == self.args.max_steps and exit == False):
            reward += self.args.step_lim_punishment
            done = True
            
        #if(verbose): print("\n\nRaw Action: x {}, y {}.".format(x, y))
        action_name = "Right" if x == 1 else "Left" if x == -1 else "Up" if y == 1 else "Down"
        if(verbose): print("\n\nStep: {}. Action: {}.".format(self.steps, action_name))
        if(verbose): print("\n{}\n".format(self))
        if(verbose): print("Reward: {}. Spot name: {}. Done: {}.".format(reward, spot_name, done))
        if(verbose): print(self.obs_str())
        if(verbose): print(self.obs())
        return(reward, spot_name, done, action_name)    
    
    def __str__(self):
        to_print = ""
        for y in self.maze.ys:
            for x in self.maze.xs:
                portrayal = " "
                for spot in self.maze.spots:
                    if(spot.pos == (x, y)): portrayal = "\u25A1"
                if(self.agent_pos == (x, y)): portrayal = "@"
                to_print += portrayal 
            if(y != self.maze.ys[-1]): to_print += "\n"
        return(to_print)
    
    
    
maze = Easy_Maze("t", args)
obs_size = maze.obs().shape[-1]
action_size = 2
    
if __name__ == "__main__":        

    print("{}\n\n{}".format(maze, maze.obs_str()))
    
    actions = [[1,0], [0,1], [-1,0]]
    for action in actions:
        reward, name, done, action_name = maze.action(action[0], action[1], verbose = True)


# %%
