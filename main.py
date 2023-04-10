#%%

import pickle, torch, random
import numpy as np
from multiprocessing import Process, Queue
from time import sleep 
from math import floor

from utils import args, folder, duration, estimate_total_duration

print("\nname:\n{}".format(args.arg_name))
print("\nagents: {}. previous_agents: {}.".format(args.agents, args.previous_agents))

if(args.hard_maze): from hard_agent import Agent
else:               from easy_agent import Agent



def train(q, i):
    seed = i + args.previous_agents
    np.random.seed(seed) ; random.seed(seed) ; torch.manual_seed(seed) ; torch.cuda.manual_seed(seed)
    agent = Agent(args)
    agent.training(q, i)
    with open(folder + "/plot_dict_{}.pickle".format(   str(i).zfill(3)), "wb") as handle:
        pickle.dump(agent.plot_dict, handle)
    with open(folder + "/min_max_dict_{}.pickle".format(str(i).zfill(3)), "wb") as handle:
        pickle.dump(agent.min_max_dict, handle)


queue = Queue()

processes = []
for worker_id in range(args.agents):
    process = Process(target=train, args=(queue, worker_id))
    processes.append(process)
    process.start()

progress_dict = {i : "0" for i in range(args.agents)}
prev_progress_dict = {i : None for i in range(args.agents)}

while any(process.is_alive() for process in processes) or not queue.empty():
    while not queue.empty():
        worker_id, progress_percentage = queue.get()
        progress_dict[worker_id] = progress_percentage

    if any(progress_dict[key] != prev_progress_dict[key] for key in progress_dict.keys()):
        prev_progress_dict = progress_dict.copy()
        string = "" ; hundreds = 0
        values = list(progress_dict.values()) ; values.sort()
        so_far = duration()
        lowest = float(values[0])
        estimated_total = estimate_total_duration(lowest)
        if(estimated_total == "?:??:??"): to_do = "?:??:??"
        else:                                   to_do = estimated_total - so_far
        values = [str(floor(100 * float(value))).ljust(3, " ") for value in values]
        for value in values:
            if(value != "100"): string += " " + value
            else:               hundreds += 1 
        if(hundreds > 0): string += " ##" + " 100" * hundreds
        string = "{} ({} left):".format(so_far, to_do) + string
        if(hundreds == 0): string += " ##"
        string = string.rstrip() + "."
        print(string, flush=True)
    sleep(1)

for process in processes:
    process.join()

print("\nDuration: {}. Done!".format(duration()))
# %%
