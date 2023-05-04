#%%
import os
import pickle
import torch
import tkinter as tk
from tkinter import ttk

from hard_maze import Hard_Maze

# To do:
#   Better selection of agent_num and epoch
#   Pause and let user change action
#   Make agent predictions

class GUI(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        
        self.plot_dict_dict = {}
        for name in [f for f in os.listdir("saved") if os.path.isdir("saved/" + f)]:
            file = "saved/{}/plot_dict.pickle".format(name)
            with open(file, "rb") as handle: plot_dict = pickle.load(handle)
            if(plot_dict["args"].hard_maze): self.plot_dict_dict[name] = plot_dict 
        saved_args = list(self.plot_dict_dict.keys())
        saved_args.sort()
        default_args = saved_args[0]
        self.argname_var = tk.StringVar()
        self.argname_var.set(default_args)
        argname_menu = ttk.OptionMenu(self, self.argname_var, default_args, *saved_args, command = self.update_epoch_agent_num)
        argname_menu.pack()
        
        self.epoch_var = tk.StringVar(value='0')
        self.epoch_menu = ttk.OptionMenu(self, self.epoch_var, '0', *["0"])
        self.epoch_menu.pack()
        
        self.agent_num_var = tk.StringVar(value='1')
        self.agent_num_menu = ttk.OptionMenu(self, self.agent_num_var, '1', *["1"])
        self.agent_num_menu.pack()
        
        maze_files = os.listdir("arenas")
        default_maze = 't.png'
        self.maze_var = tk.StringVar()
        self.maze_var.set(default_maze)
        maze_menu = ttk.OptionMenu(self, self.maze_var, default_maze, *maze_files)
        maze_menu.pack()
        
        run_button = tk.Button(self, text="Run", command=self.run)
        run_button.pack()
        
        self.update_epoch_agent_num()
        
    def update_plot_dict(self):
        self.plot_dict = self.plot_dict_dict[self.argname_var.get()]
        
    def update_epoch_agent_num(self, *args):
        self.update_plot_dict()
        agent_names = list(self.plot_dict["agent_lists"].keys())[3:]
        agent_nums = sorted(set([s.split('_')[0] for s in agent_names]), key = int)
        epochs = sorted(set([s.split('_')[1] for s in agent_names]), key = int)
                        
        self.agent_num_menu['menu'].delete(0, 'end')
        for agent_num in agent_nums:
            self.agent_num_menu['menu'].add_command(label=agent_num, command=lambda num=agent_num: self.agent_num_var.set(num))
        
        self.epoch_menu['menu'].delete(0, 'end')
        for epoch in epochs:
            self.epoch_menu['menu'].add_command(label=epoch, command=lambda ep=epoch: self.epoch_var.set(ep))
            
            
        
    def run(self):
        agent_name = "{}_{}".format(self.agent_num_var.get(), self.epoch_var.get())
        print("\narg name: {}.\nAgent num: {}. Epoch: {}.\nMaze: {}.\n".format(
            self.plot_dict["arg_name"], self.agent_num_var.get(), self.epoch_var.get(), self.maze_var.get()))
        args = self.plot_dict["args"]
        agent_lists = self.plot_dict["agent_lists"]
        state_dict = agent_lists[agent_name]
        
        forward        = agent_lists["forward"](args)
        actor          = agent_lists["actor"](args)
        critic1        = agent_lists["critic"](args)
        critic1_target = agent_lists["critic"](args)
        critic2        = agent_lists["critic"](args)
        critic2_target = agent_lists["critic"](args)
        
        forward.load_state_dict(       state_dict[0])
        actor.load_state_dict(         state_dict[1])
        critic1.load_state_dict(       state_dict[2])
        critic1_target.load_state_dict(state_dict[3])
        critic2.load_state_dict(       state_dict[4])
        critic2_target.load_state_dict(state_dict[5])
                
        maze = Hard_Maze(self.maze_var.get(), True, actor.args)
        done = False
        prev_a = torch.zeros((1, 1, 2))
        h_actor = torch.zeros((1, 1, actor.args.hidden_size))
        while(not done):
            o, s = maze.obs()
            a, _, h_actor = actor(rgbd = o, spe = s, prev_action = prev_a, h = h_actor)
            action = torch.flatten(a).tolist()
            _, _, done, _ = maze.action(action[0], action[1], True)
            prev_a = a
        maze.maze.stop()
        


if __name__ == "__main__":
    root = tk.Tk()
    GUI(root).pack(fill="both", expand=True)
    root.mainloop()
# %%
