#%%
from random import choices
import pandas as pd
import numpy as np
import pybullet as p
import cv2
from itertools import product
from math import pi, sin, cos

from utils import default_args, args



class Exit:
    def __init__(self, name, pos, rew):     # Position (Y, X)
        self.name = name ; self.pos = pos ; self.rew = rew
        
class Arena_Dict:
    def __init__(self, start, exits):
        self.start = start 
        self.exits = pd.DataFrame(
            data = [[exit.name, exit.pos, exit.rew] for exit in exits],
            columns = ["Name", "Position", "Reward"])
        
arena_dict = {
    "t.png" : Arena_Dict(
        (3, 2),
        [Exit(  "L",    (2,0), args.default_reward),
        Exit(   "R",    (2,7), args.better_reward)]),
    "1.png" : Arena_Dict(
        (2,2), 
        [Exit(  "L",    (1,0), args.default_reward),
        Exit(   "R",    (1,4), args.better_reward)]),
    "2.png" : Arena_Dict(
        (3,3), 
        [Exit(  "LL",   (4,1), args.better_reward),
        Exit(   "LR",   (0,1), args.default_reward),
        Exit(   "RL",   (0,5), args.default_reward),
        Exit(   "RR",   (4,5), args.default_reward)]),
    "3.png" : Arena_Dict(
        (4,4), 
        [Exit(  "LLL",  (6,3), args.default_reward),
        Exit(   "LLR",  (6,1), args.default_reward),
        Exit(   "LRL",  (0,1), args.default_reward),
        Exit(   "LRR",  (0,3), args.default_reward),
        Exit(   "RLL",  (0,5), args.better_reward),
        Exit(   "RLR",  (0,7), args.default_reward),
        Exit(   "RRL",  (6,7), args.default_reward),
        Exit(   "RRR",  (6,5), args.default_reward)])}



def get_physics(GUI, w, h):
    if(GUI):
        physicsClient = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(1,90,-89,(w/2,h/2,w), physicsClientId = physicsClient)
    else:   
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath("pybullet_data/")
    return(physicsClient)

def enable_opengl():
    import pkgutil
    egl = pkgutil.get_loader('eglRenderer')
    import pybullet_data

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
    # print("plugin=", plugin)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)



class Arena():
    def __init__(self, arena_name, args = args, GUI = False):
        #enable_opengl()
        self.args = args
        self.arena_name = arena_name
        self.start = arena_dict[arena_name + ".png"].start
        self.exits = arena_dict[arena_name + ".png"].exits
        self.arena_map = cv2.imread("arenas/" + arena_name + ".png")
        self.w, self.h, _ = self.arena_map.shape
        self.physicsClient = get_physics(GUI, self.w, self.h)
        self.ends = {}
        self.colors = {}
        self.already_constructed = False

    def start_arena(self):
        if(not self.already_constructed):
            p.loadURDF("plane.urdf", [0,0,0], globalScaling = .5,
                       useFixedBase = True, physicsClientId = self.physicsClient) 
            p.loadURDF("plane.urdf", [10,0,0], globalScaling = .5,
                       useFixedBase = True, physicsClientId = self.physicsClient) 
            p.loadURDF("plane.urdf", [0,10,0], globalScaling = .5,
                       useFixedBase = True, physicsClientId = self.physicsClient) 
            p.loadURDF("plane.urdf", [10,10,0], globalScaling = .5,
                       useFixedBase = True, physicsClientId = self.physicsClient) 
            self.ends = {}
            for loc in ((x,y) for x in range(self.w) for y in range(self.h)):
                pos = [loc[0],loc[1],.5]
                if((self.arena_map[loc] == [255]).all()):
                    if(not self.exits.loc[self.exits["Position"] == loc].empty):
                        row = self.exits.loc[self.exits["Position"] == loc]
                        end_pos = ((pos[0]-.5, pos[0] + .5), (pos[1] - .5, pos[1] + .5))
                        self.ends[row["Name"].values[0]] = (end_pos, row["Reward"].values[0])
                else:
                    ors = p.getQuaternionFromEuler([0,0,0])
                    color = self.arena_map[loc][::-1] / 255
                    color = np.append(color, 1)
                    cube_size = 1/self.args.boxes_per_cube
                    cubes = [p.loadURDF("cube.urdf", (pos[0]+i*cube_size, pos[1]+j*cube_size, pos[2]+k*cube_size), 
                                    ors, globalScaling = cube_size, useFixedBase = True, physicsClientId = self.physicsClient) \
                                        for i, j, k in product([l/2 for l in range(-self.args.boxes_per_cube+1, self.args.boxes_per_cube+1, 2)], repeat=3)]
                    bigger_cube = p.loadURDF("cube.urdf", pos, ors, globalScaling = self.args.bigger_cube,
                                    useFixedBase = True, 
                                    physicsClientId = self.physicsClient)
                    self.colors[bigger_cube] = (0,0,0,0)
                    for cube in cubes:
                        self.colors[cube] = color
            self.already_constructed = True
            
            self.colorize()
            #p.saveWorld("arenas/" + self.args.arena_name + ".urdf")
                
        inherent_roll = pi/2
        inherent_pitch = 0
        yaw = 0
        spe = self.args.min_speed
        color = [1,0,0,1]
        file = "ted_duck.urdf"
        
        pos = (self.start[0], self.start[1], .5)
        orn = p.getQuaternionFromEuler([inherent_roll,inherent_pitch,yaw])
        num = p.loadURDF(file,pos,orn,
                           globalScaling = self.args.body_size, 
                           physicsClientId = self.physicsClient)
        p.changeDynamics(num, 0, maxJointVelocity=10000)
        x, y = cos(yaw)*spe, sin(yaw)*spe
        p.resetBaseVelocity(num, (x,y,0),(0,0,0), physicsClientId = self.physicsClient)
        p.changeVisualShape(num, -1, rgbaColor = color, physicsClientId = self.physicsClient)
                    
    def colorize(self):
        for cube, color in self.colors.items():
            p.changeVisualShape(cube, -1, rgbaColor = color, physicsClientId = self.physicsClient)
        
    def get_pos_yaw_spe(self, num):
        pos, ors = p.getBasePositionAndOrientation(num, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors)[-1]
        (x, y, _), _ = p.getBaseVelocity(num, physicsClientId = self.physicsClient)
        spe = (x**2 + y**2)**.5
        return(pos, yaw, spe)
    
    def pos_in_box(self, num, box):
        (min_x, max_x), (min_y, max_y) = box 
        pos, _, _ = self.get_pos_yaw_spe(num)
        in_x = pos[0] >= min_x and pos[0] <= max_x 
        in_y = pos[1] >= min_y and pos[1] <= max_y 
        return(in_x and in_y)
    
    def end_collisions(self, num):
        col = False
        which = ("FAIL", -1)
        reward = 0
        for end_name, (end, end_reward) in self.ends.items():
            if self.pos_in_box(num, end):
                col = True
                which = (end_name, end_reward)
                reward = end_reward
        if(type(reward) != tuple): pass
        else:
            weights = [w for w, r in reward]
            reward_index = choices([i for i in range(len(reward))], weights = weights)[0]
            reward = reward[reward_index][1]
        return(col, which, reward)
    
    def other_collisions(self, num):
        col = False
        for cube in self.colors.keys():
            if 0 < len(p.getContactPoints(num, cube, physicsClientId = self.physicsClient)):
                col = True
        return(col)



if __name__ == "__main__":
    arena = Arena(arena_name = "3", GUI = True)
    arena.start_arena()


# %%
