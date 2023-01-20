#%%

from utils import args
from train import Trainer

bayes = True
dkl_rate = .001
entropy = -2
curiosity = 5
naive = True 

args.bayes = bayes
args.dkl_rate = dkl_rate
if(entropy != False): args.alpha = None ; args.target_entropy = entropy
if(curiosity != False):
    if(not naive): args.naive_curiosity = False 
    args.eta = curiosity
    
title = "plots_{}_dkl_rate_{}_entropy_{}{}_curiosity{}".format(
    dkl_rate, 
    entropy if entropy != False else "no",
    "naive_" if naive and curiosity != False else "",
    curiosity if curiosity != False else "no", 
    "_bayes" if bayes else "")

print(title)

trainer = Trainer(args, title)
trainer.train()

print("\n\nFinished training!")

# %%
