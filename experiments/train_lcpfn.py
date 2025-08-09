from functools import partial

import torch
import numpy as np
import lcpfn
import matplotlib.pyplot as plt

from nosbench.program import *
from nosbench.function import *

from nosbench.device import Device

Device.set("cuda")

program = Program(
 [Instruction(Function(torch.square, 1), inputs=[17], output=11),
 Instruction(Function(torch.sub, 2), inputs=[3, 5], output=9),
 Instruction(Function(torch.sub, 2), inputs=[3, 7], output=10),
 Instruction(Function(interpolate, 3), inputs=[12, 1, 9], output=12),
 Instruction(Function(interpolate, 3), inputs=[13, 11, 10], output=13),
 Instruction(Function(bias_correct, 3), inputs=[12, 6, 3], output=14),
 Instruction(Function(bias_correct, 3), inputs=[13, 6, 2], output=15),
 Instruction(Function(torch.maximum, 2), inputs=[7, 0], output=16),
 Instruction(Function(torch.sqrt, 1), inputs=[15], output=17),
 Instruction(Function(torch.add, 2), inputs=[17, 16], output=17),
 Instruction(Function(torch.div, 2), inputs=[14, 17], output=19),
 Instruction(Function(torch.mul, 2), inputs=[19, 7], output=16)])


get_batch_func = lcpfn.create_get_batch_func(prior=lcpfn.sample_from_prior)
X, Y, Y_noisy = get_batch_func(batch_size=100, seq_len=100, num_features=1)
print(X.shape, Y.shape, Y_noisy.shape)

# optimizer = partial(torch.optim.AdamW, lr=0.001, weight_decay=0.0)
optimizer = program.optimizer()

result = lcpfn.train_lcpfn(get_batch_func=get_batch_func, 
                         optimizer=optimizer,
                         seq_len=100,
                         emsize=256,
                         nlayers=12,
                         num_borders=1000,
                         lr=0.001, # Not USED
                         batch_size=10,
                        scheduler=None,
                         epochs=20)

model = lcpfn.LCPFN.from_model(result[2])

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

plt.savefig("lcpfn_inference.png")
