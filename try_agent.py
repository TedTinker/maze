#%%
import os
import pickle
import torch
import tkinter as tk
from tkinter import ttk
from math import degrees, radians
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from hard_maze import Hard_Maze

# To do:
#   Only saving last epoch?
#   Make predictions for rewards, too

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
        self.argname_label = tk.Label(self, text="Arg Name:")
        self.argname_label.grid(row=0, column=0, sticky="e")
        argname_menu = ttk.OptionMenu(self, self.argname_var, default_args, *saved_args, command=self.update_epoch_agent_num)
        argname_menu.grid(row=0, column=1)
        
        self.epoch_var = tk.StringVar(value='0')
        self.epoch_menu = ttk.OptionMenu(self, self.epoch_var, '0', *["0"])
        self.epoch_label = tk.Label(self, text="Epoch:")
        
        self.agent_num_var = tk.StringVar(value='1')
        self.agent_num_menu = ttk.OptionMenu(self, self.agent_num_var, '1', *["1"])
        self.agent_num_label = tk.Label(self, text="Agent Number:")
        
        maze_files = os.listdir("arenas")
        default_maze = 't.png'
        self.maze_var = tk.StringVar()
        self.maze_var.set(default_maze)
        maze_menu = ttk.OptionMenu(self, self.maze_var, default_maze, *maze_files)
        self.maze_label = tk.Label(self, text="Maze:")
        
        step_button = tk.Button(self, text="Step", command=self.step)
        
        self.yaw_label = tk.Label(self, text="Yaw:")
        self.yaw_var = tk.StringVar()
        self.yaw_entry = tk.Entry(self, textvariable=self.yaw_var)
        
        self.speed_label = tk.Label(self, text="Speed:")
        self.speed_var = tk.StringVar()
        self.speed_entry = tk.Entry(self, textvariable=self.speed_var)
        
        predict_button = tk.Button(self, text="Predict", command=self.predict)
        
        self.fig_1, self.ax_1 = plt.subplots(figsize=(5, 4))
        self.ax_1.plot([0], [0])
        self.ax_1.axis("off")
        self.plot_canvas_1 = FigureCanvasTkAgg(self.fig_1, master=self)
        self.plot_canvas_1.draw()
        
        self.fig_2, self.ax_2 = plt.subplots(figsize=(5, 4))
        self.ax_2.plot([0], [0])
        self.ax_2.axis("off")
        self.plot_canvas_2 = FigureCanvasTkAgg(self.fig_2, master=self)
        self.plot_canvas_2.draw()
                
        self.epoch_label.grid(      row=1, column=0, sticky="e")
        self.epoch_menu.grid(       row=1, column=1)
        self.agent_num_label.grid(  row=2, column=0, sticky="e")
        self.agent_num_menu.grid(   row=2, column=1)
        self.maze_label.grid(       row=3, column=0, sticky="e")
        maze_menu.grid(             row=3, column=1)
        step_button.grid(           row=4, column=0, columnspan=2)
        self.yaw_label.grid(        row=5, column=0, sticky="e")
        self.yaw_entry.grid(        row=5, column=1)
        self.speed_entry.grid(      row=6, column=1)
        self.speed_label.grid(      row=6, column=0, sticky="e")
        predict_button.grid(        row=7, column=0, columnspan=2)
        self.plot_canvas_1.get_tk_widget().grid(row=8, column=0, columnspan=2, padx=5, pady=5)
        self.plot_canvas_2.get_tk_widget().grid(row=8, column=6, columnspan=2, padx=5, pady=5)
        
        self.update_epoch_agent_num()
        
        self.done = True
        
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
            
            
            

    def add_agent_action(self, yaw, spe):
        yaw = yaw * self.actor.args.max_yaw_change
        yaw = [-self.actor.args.max_yaw_change, self.actor.args.max_yaw_change, yaw] ; yaw.sort() ; yaw = yaw[1]
        yaw = round(degrees(yaw))
        spe = self.actor.args.min_speed + ((spe + 1)/2) * \
            (self.actor.args.max_speed - self.actor.args.min_speed)
        spe = [self.actor.args.min_speed, self.actor.args.max_speed, round(spe)] ; spe.sort() ; spe = spe[1]
        self.yaw_var.set(str(yaw))
        self.speed_var.set(str(spe))
        
    def get_real_action(self):
        yaw = float(self.yaw_var.get())
        spe = float(self.speed_var.get())
        yaw = radians(yaw) / self.actor.args.max_yaw_change
        speed = (((spe - self.actor.args.min_speed) / (self.actor.args.max_speed - self.actor.args.min_speed)) * 2) - 1
        return(yaw, speed)
            
            
        
    def step(self):
        if(self.done):
            agent_name = "{}_{}".format(self.agent_num_var.get(), self.epoch_var.get())
            print("\narg name: {}.\nAgent num: {}. Epoch: {}.\nMaze: {}.\n".format(
                self.plot_dict["arg_name"], self.agent_num_var.get(), self.epoch_var.get(), self.maze_var.get()))
            args = self.plot_dict["args"]
            agent_lists = self.plot_dict["agent_lists"]
            state_dict = agent_lists[agent_name]
            
            self.forward        = agent_lists["forward"](args)
            self.actor          = agent_lists["actor"](args)
            self.critic1        = agent_lists["critic"](args)
            self.critic1_target = agent_lists["critic"](args)
            self.critic2        = agent_lists["critic"](args)
            self.critic2_target = agent_lists["critic"](args)
            
            self.forward.load_state_dict(       state_dict[0])
            self.actor.load_state_dict(         state_dict[1])
            self.critic1.load_state_dict(       state_dict[2])
            self.critic1_target.load_state_dict(state_dict[3])
            self.critic2.load_state_dict(       state_dict[4])
            self.critic2_target.load_state_dict(state_dict[5])
                    
            self.maze = Hard_Maze(self.maze_var.get(), True, self.actor.args)
            o, s = self.maze.obs()
            self.done = False
            self.prev_a   = torch.zeros((1, 1, 2))
            self.h_q      = torch.zeros((1, 1, self.actor.args.hidden_size))
            self.h_actor  = torch.zeros((1, 1, self.actor.args.hidden_size))
            self.h_critic = torch.zeros((1, 1, self.actor.args.hidden_size))
            
            _, (self.zq_mu, self.zq_std), self.h_q = self.forward(o, s, self.prev_a, self.h_q)
            
            a, _, self.h_actor = self.actor(o, s, self.prev_a, self.h_actor)
            yaw, speed = torch.flatten(a).tolist()
            self.add_agent_action(yaw, speed)
            
            self.Qs = []
                        
        else:
            yaw, speed = self.get_real_action()
            _, _, self.done, _ = self.maze.action(yaw, speed, True)
            self.prev_a = torch.tensor([[[yaw, speed]]])
            o, s = self.maze.obs()
            _, (self.zq_mu, self.zq_std), self.h_q = self.forward(o, s, self.prev_a, self.h_q)
            a, _, self.h_actor = self.actor(o, s, self.prev_a, self.h_actor)
            Q, self.h_critic = self.critic1(o, s, a, self.h_critic)
            self.Qs.append(Q.item())
            self.plot_Qs()
            yaw, speed = torch.flatten(a).tolist()
            self.add_agent_action(yaw, speed)
            if(self.done): self.maze.maze.stop()
            
    def predict(self):
        yaw, speed = self.get_real_action()
        action = torch.tensor([[[yaw, speed]]])
        with(torch.no_grad()):
            (_, zq_preds_rgbd), (_, zq_preds_spe) = self.forward.get_preds(action, self.zq_mu, self.zq_std, self.h_q, quantity = 1)
        pred_rgbd = zq_preds_rgbd[0].squeeze(0).squeeze(0)
        pred_spe  = zq_preds_spe[0].squeeze(0).squeeze(0)
        
        self.ax_1.clear()
        self.ax_1.axis("off")
        self.ax_1.set_title("Speed {}".format(round(pred_spe.item())))
        self.ax_1.imshow(torch.sigmoid(pred_rgbd[:,:,0:3])) 
        self.plot_canvas_1.draw()
        
    def plot_Qs(self):
        self.ax_2.clear() 
        self.ax_2.plot([i for i in range(len(self.Qs))], self.Qs)
        self.ax_2.set_title("Qs")
        self.plot_canvas_2.draw()
        


if __name__ == "__main__":
    root = tk.Tk()
    GUI(root).pack(fill="both", expand=True)
    root.mainloop()
# %%
