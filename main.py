#%%

import pickle

from utils import args, folder
from train import Trainer

print("name:", args.explore_type + "_" + str(args.id))

import datetime 
start_time = datetime.datetime.now()

def reset_start_time():
    global start_time
    start_time = datetime.datetime.now()
    
def duration():
    global start_time
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)


trainer = Trainer(args, "{}_{}".format(args.explore_type, args.id))
plot_dict, min_max_dict = trainer.train()

with open(folder + "/plot_dict_{}.pickle".format(   str(args.id).zfill(3)), "wb") as handle:
    pickle.dump(plot_dict, handle)
with open(folder + "/min_max_dict_{}.pickle".format(str(args.id).zfill(3)), "wb") as handle:
    pickle.dump(min_max_dict, handle)
    
print("Duration: {}".format(duration()))

# %%
