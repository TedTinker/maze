#%%

from utils import args
from train import Trainer

bayes = False
entropy = False 
curiosity = False 
naive = True 

args.bayes = bayes
if(entropy != False): args.alpha = None ; args.target_entropy = entropy
if(curiosity != False):
    if(not naive): args.naive_curiosity = False 
    args.eta = curiosity
    
folder = "plots_{}_entropy_{}{}_curiosity{}".format(
    entropy if entropy != False else "no",
    "naive_" if naive and curiosity != False else "",
    curiosity if curiosity != False else "no", 
    "_bayes" if bayes else "")

trainer = Trainer(args, folder)
trainer.train()

print("\n\nFinished training!")

# %%
