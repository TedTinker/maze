#%%

from utils import args
from train import Trainer

trainer = Trainer(args)
trainer.train()

print("\n\nFinished training!")

# %%
