#%%
import numpy as np
import pybullet as p
from math import pi, degrees, sin, cos

from utils import default_args, args, print
from arena import Arena



import torch
from torchvision.transforms.functional import resize

class Hard_Maze:
    
    def __init__(self, arena_name, GUI = False, args = default_args):
        self.args = args
        self.maze = Arena(arena_name, GUI, args)
        self.begin()
        
    def begin(self):
        self.steps = 0 
        self.maze.begin()
        self.agent_pos, self.agent_yaw, self.agent_spe = self.maze.get_pos_yaw_spe()
        
    def obs(self):
        x, y = cos(self.agent_yaw), sin(self.agent_yaw)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition = [self.agent_pos[0], self.agent_pos[1], .4], 
            cameraTargetPosition = [self.agent_pos[0] - x, self.agent_pos[1] - y, .4], 
            cameraUpVector = [0, 0, 1], physicsClientId = self.maze.physicsClient)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov = 90, aspect = 1, nearVal = .01, 
            farVal = 10, physicsClientId = self.maze.physicsClient)
        _, _, rgba, depth, _ = p.getCameraImage(
            width=32, height=32,
            projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow = 0,
            physicsClientId = self.maze.physicsClient)
        
        rgb = np.divide(rgba[:,:,:-1], 255)
        d = np.nan_to_num(np.expand_dims(depth, axis=-1), nan=1)
        if(d.max() == d.min()): pass
        else: d = (d.max() - d)/(d.max()-d.min())
        rgbd = np.concatenate([rgb, d], axis = -1)
        rgbd = torch.from_numpy(rgbd).float()
        rgbd = resize(rgbd.permute(-1,0,1), (self.args.image_size, self.args.image_size)).permute(1,2,0)
        spe = torch.tensor(self.agent_spe).unsqueeze(0)
        return(rgbd.unsqueeze(0), spe.unsqueeze(0))
    
    def change_velocity(self, yaw_change, speed, verbose = False):
        old_yaw = self.agent_yaw
        new_yaw = old_yaw + yaw_change
        new_yaw %= 2*pi
        orn = p.getQuaternionFromEuler([pi/2, 0, new_yaw])
        self.maze.resetBasePositionAndOrientation((self.agent_pos[0], self.agent_pos[1], .5), orn)
        
        old_speed = self.agent_spe
        x = -cos(new_yaw)*speed / self.args.steps_per_step
        y = -sin(new_yaw)*speed / self.args.steps_per_step
        self.maze.resetBaseVelocity(x, y)
        _, self.agent_yaw, _ = self.maze.get_pos_yaw_spe()
                
        if(verbose):
            print("\n\nOld yaw:\t{}\nChange:\t\t{}\nNew yaw:\t{}".format(
                round(degrees(old_yaw)) % 360, round(degrees(yaw_change)), round(degrees(new_yaw))))
            print("Old speed:\t{}\nNew speed:\t{}".format(old_speed, speed))
            #self.render(view = "body")  
            print("\n")
        
    def action(self, yaw, spe, verbose = True):
        self.steps += 1
        
        if(verbose): print("\n\nStep {}: yaw {}, spe {}.".format(self.steps, yaw, spe))
        yaw = -yaw * self.args.max_yaw_change
        yaw = [-self.args.max_yaw_change, self.args.max_yaw_change, yaw] ; yaw.sort() ; yaw = yaw[1]
        spe = self.args.min_speed + ((spe + 1)/2) * \
            (self.args.max_speed - self.args.min_speed)
        spe = [self.args.min_speed, self.args.max_speed, spe] ; spe.sort() ; spe = spe[1]
        if(verbose): print("updated: yaw {}, spe {}.".format(yaw, spe))
            
        self.change_velocity(yaw, spe, verbose = verbose)
        for _ in range(self.args.steps_per_step):
            p.stepSimulation(physicsClientId = self.maze.physicsClient)
            
        self.agent_pos, self.agent_yaw, self.agent_spe = self.maze.get_pos_yaw_spe()
        if(verbose): print("agent: pos {}, yaw {}, spe {}.".format(self.agent_pos, self.agent_yaw, self.agent_spe))
        
        end, which, reward = self.maze.end_collisions()
        if(verbose): print("end {}, which {}, reward {}".format(end, which, reward))
        
        col = self.maze.other_collisions()
        if(col): reward -= self.args.wall_punishment
        if(not end): end = self.steps >= self.args.max_steps
        exit = which != "NONE"
        if(end and not exit): reward = self.args.step_lim_punishment
        if(verbose): print("end {}, which {}, reward {}\n\n".format(end, which, reward))

        return(reward, which, end)
    
    
    
if __name__ == "__main__":        
    from random import random
    from time import sleep
    import matplotlib.pyplot as plt

    maze = Hard_Maze("t", True, args)
    done = False
    while(done == False):
        reward, name, done = maze.action(random(), random(), verbose = True)
        rgbd, spe = maze.obs()
        plt.imshow(rgbd[:,:,0:3])
        plt.show()
        plt.close()
        sleep(1)
# %%
