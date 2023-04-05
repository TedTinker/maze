#%%

import pickle, torch, random
import numpy as np
from multiprocessing import Process, Queue
from time import sleep 

from utils import args, folder, duration
from agent import Agent

print("\nname:\n{}".format(args.arg_name))
print("\nagents: {}. previous_agents: {}.".format(args.agents, args.previous_agents))



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

progress_dict = {i : "0%" for i in range(args.agents)}
prev_progress_dict = {i : None for i in range(args.agents)}

while any(process.is_alive() for process in processes) or not queue.empty():
    while not queue.empty():
        worker_id, progress_percentage = queue.get()
        progress_dict[worker_id] = progress_percentage

    if any(progress_dict[key] != prev_progress_dict[key] for key in progress_dict.keys()):
        prev_progress_dict = progress_dict.copy()
        string = ""
        for key, item in progress_dict.items():
            if(item != "100%"): string += " " + item
        string = "Duration: {} / {}.".format(duration(), "estimate") + string
        print(string, flush=True)
    sleep(1)

for process in processes:
    process.join()

print("Done with {}!".format(args.arg_name))
print("\n\nDuration: {}".format(duration()))
# %%
