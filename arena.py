#%%
from random import choices
import pandas as pd
import numpy as np
import pybullet as p
import cv2, os
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
        Exit(   "R",    (2,4), args.better_reward)]),
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
    def __init__(self, arena_name, GUI = False, args = default_args):
        #enable_opengl()
        self.args = args
        self.start = arena_dict[arena_name + ".png"].start
        self.exits = arena_dict[arena_name + ".png"].exits
        arena_map = cv2.imread("arenas/" + arena_name + ".png")
        w, h, _ = arena_map.shape
        self.physicsClient = get_physics(GUI, w, h)

        plane_positions = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]]
        plane_ids = []
        for position in plane_positions:
            plane_id = p.loadURDF("plane.urdf", position, globalScaling=.5, useFixedBase=True, physicsClientId=self.physicsClient)
            plane_ids.append(plane_id)

        self.ends = {} ; self.colors = {} 
        for loc in ((x, y) for x in range(w) for y in range(h)):
            pos = [loc[0], loc[1], .5]
            if ((arena_map[loc] == [255]).all()):
                if (not self.exits.loc[self.exits["Position"] == loc].empty):
                    row = self.exits.loc[self.exits["Position"] == loc]
                    end_pos = ((pos[0] - .5, pos[0] + .5), (pos[1] - .5, pos[1] + .5))
                    self.ends[row["Name"].values[0]] = (end_pos, row["Reward"].values[0])
            else:
                ors = p.getQuaternionFromEuler([0, 0, 0])
                color = arena_map[loc][::-1] / 255
                color = np.append(color, 1)
                cube = p.loadURDF("cube.urdf", (pos[0], pos[1], pos[2]), ors, 
                                  useFixedBase=True, physicsClientId=self.physicsClient)
                self.colors[cube] = color
        
        for cube, color in self.colors.items():
            p.changeVisualShape(cube, -1, rgbaColor = color, physicsClientId = self.physicsClient)

        inherent_roll = pi/2
        inherent_pitch = 0
        yaw = 0
        spe = self.args.min_speed
        color = [1,0,0,1]
        file = "ted_duck.urdf"
        pos = (self.start[0], self.start[1], .5)
        orn = p.getQuaternionFromEuler([inherent_roll, inherent_pitch, yaw])
        self.body_num = p.loadURDF(file, pos, orn,
                           globalScaling = self.args.body_size, 
                           physicsClientId = self.physicsClient)
        p.changeDynamics(self.body_num, 0, maxJointVelocity=10000)
        x, y = cos(yaw)*spe, sin(yaw)*spe
        self.resetBaseVelocity(x, y)
        p.changeVisualShape(self.body_num, -1, rgbaColor = color, physicsClientId = self.physicsClient)
        
    def begin(self):
        inherent_roll = pi/2
        inherent_pitch = 0
        yaw = 0
        spe = self.args.min_speed
        pos = (self.start[0], self.start[1], .5)
        orn = p.getQuaternionFromEuler([inherent_roll, inherent_pitch, yaw])
        x, y = cos(yaw)*spe, sin(yaw)*spe
        self.resetBaseVelocity(x, y)
        self.resetBasePositionAndOrientation(pos, orn)
        
        
    def get_pos_yaw_spe(self):
        pos, ors = p.getBasePositionAndOrientation(self.body_num, physicsClientId = self.physicsClient)
        yaw = p.getEulerFromQuaternion(ors)[-1]
        (x, y, _), _ = p.getBaseVelocity(self.body_num, physicsClientId = self.physicsClient)
        spe = (x**2 + y**2)**.5
        return(pos, yaw, spe)
    
    def resetBasePositionAndOrientation(self, pos, orn):
        p.resetBasePositionAndOrientation(self.body_num, pos, orn, physicsClientId = self.physicsClient)
        
    def resetBaseVelocity(self, x, y):    
        p.resetBaseVelocity(self.body_num, (x,y,0), (0,0,0), physicsClientId = self.physicsClient)
    
    def pos_in_box(self, box):
        (min_x, max_x), (min_y, max_y) = box 
        pos, _, _ = self.get_pos_yaw_spe()
        in_x = pos[0] >= min_x and pos[0] <= max_x 
        in_y = pos[1] >= min_y and pos[1] <= max_y 
        return(in_x and in_y)
    
    def end_collisions(self):
        col = False
        which = "NONE"
        reward = ((1, 0),)
        for end_name, (end, end_reward) in self.ends.items():
            if self.pos_in_box(end):
                col = True
                which = end_name
                reward = end_reward
        weights = [w for w, r in reward]
        rewards = [r for w, r in reward]
        reward = choices(rewards, weights = weights, k = 1)[0]
        return(col, which, reward)
    
    def other_collisions(self):
        col = False
        for cube in self.colors.keys():
            if 0 < len(p.getContactPoints(self.body_num, cube, physicsClientId = self.physicsClient)):
                col = True
        return(col)
    
    def stop(self):
        p.disconnect(self.physicsClient)

# %%
