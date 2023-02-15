#%%

import pickle

from utils import args, folder
from train import Trainer


trainer = Trainer(args, "{}_{}".format(args.explore_type, args.id))
plot_dict, min_max_dict = trainer.train()

with open(folder + "/plot_dict_{}.pickle".format(   str(args.id).zfill(3)), "wb") as handle:
    pickle.dump(plot_dict, handle)
with open(folder + "/min_max_dict_{}.pickle".format(str(args.id).zfill(3)), "wb") as handle:
    pickle.dump(min_max_dict, handle)

# %%
