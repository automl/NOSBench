from functools import partial

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import *

import lcpfn

get_batch_func = lcpfn.create_get_batch_func(prior=lcpfn.sample_from_prior)
X, Y, Y_noisy = get_batch_func(batch_size=100, seq_len=100, num_features=1)
print(X.shape, Y.shape, Y_noisy.shape)

optimizer = partial(torch.optim.AdamW, lr=0.0001, weight_decay=0.0)

if True:
 result = lcpfn.train_lcpfn(get_batch_func=get_batch_func, 
                          optimizer=optimizer,
                          seq_len=100,
                          emsize=512,
                          nlayers=12,
                          num_borders=1000,
                          lr=0.0001, # Not USED
                          batch_size=10,
                          # scheduler=None,
                          epochs=1000)

 torch.save(result[2], "lcpfn_model_adamw.pt")
 torch.save(result[4], "lcpfn_results_adamw.pt")

 model = lcpfn.LCPFN.from_model(result[2])
else:
 model = lcpfn.LCPFN.from_model(torch.load("lcpfn_model_adamw.pt"))

prior = lcpfn.sample_from_prior(np.random)
curve, _ = prior()

x = torch.arange(1, 101).unsqueeze(1)
y = torch.from_numpy(curve).float().unsqueeze(1)
cutoff = 10

predictions = model.predict_quantiles(x_train=x[:cutoff], y_train=y[:cutoff], x_test=x[cutoff:], qs=[0.05, 0.5, 0.95])

# plot data
plt.plot(curve, "black", label="target")

# plot extrapolation
plt.plot(x[cutoff:], predictions[:, 1], "blue", label="Extrapolation by PFN")
plt.fill_between(
        x[cutoff:].flatten(), predictions[:, 0], predictions[:, 2], color="blue", alpha=0.2, label="CI of 90%"
)

# plot cutoff
plt.vlines(cutoff, 0, 1, linewidth=0.5, color="k", label="cutoff")
plt.ylim(0, 1)
plt.legend(loc="lower right")

plt.savefig("lcpfn_inference_adamw.png")

