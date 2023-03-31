#%%

import pickle, os
from multiprocessing import Pool

from utils import args, folder, duration
from agent import Agent

print("name:\n{}".format(args.name))



def train(i):
    print("\n\nTraining {}!".format(i))
    agent = Agent(args)
    plot_dict, min_max_dict = agent.training(str(i).zfill(3))
    with open(folder + "/plot_dict_{}.pickle".format(   str(i).zfill(3)), "wb") as handle:
        pickle.dump(plot_dict, handle)
    with open(folder + "/min_max_dict_{}.pickle".format(str(i).zfill(3)), "wb") as handle:
        pickle.dump(min_max_dict, handle)

with Pool(5) as p: 
    p.map(train, range(1, args.agents + 1))
    p.close() ; p.join()
print("\n\nDuration: {}".format(duration()))



plot_dict = {} ; min_max_dict = {}
files = os.listdir(folder) ; files.sort()

for file in files:
    if(file.split("_")[0] == "plot"): d = plot_dict
    if(file.split("_")[0] == "min"):  d = min_max_dict
    with open(folder + "/" + file, "rb") as handle: 
        saved_d = pickle.load(handle) ; os.remove(folder + "/" + file)
    for key in saved_d.keys(): 
        if(not key in d): d[key] = []
        d[key].append(saved_d[key])
d["title"] = args.name
    
for key in min_max_dict.keys():
    if(not key in ["args", "title", "spot_names"]):
        minimum = None ; maximum = None
        for min_max in min_max_dict[key]:
            if(minimum == None):        minimum = min_max[0]
            elif(minimum > min_max[0]): minimum = min_max[0]
            if(maximum == None):        maximum = min_max[1]
            elif(maximum < min_max[1]): maximum = min_max[1]
        min_max_dict[key] = (minimum, maximum)

with open(folder + "/plot_dict.pickle", "wb") as handle:
    pickle.dump(plot_dict, handle)
with open(folder + "/min_max_dict.pickle", "wb") as handle:
    pickle.dump(min_max_dict, handle)
        
print("Done with {}!".format(args.name))

# %%
