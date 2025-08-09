import matplotlib.pyplot as plt
from torch import *
import numpy as np

import nosbench
from nosbench.program import *
from nosbench.function import *
from nosbench.optimizers import *

Incumbent = Program(
    [
        Instruction(Function(torch.sub, 2), inputs=[13, 1], output=17),
        Instruction(Function(torch.square, 1), inputs=[17], output=11),
        Instruction(Function(torch.sub, 2), inputs=[11, 8], output=13),
        Instruction(Function(torch.add, 2), inputs=[1, 13], output=13),
        Instruction(Function(torch.sub, 2), inputs=[3, 5], output=10),
        Instruction(Function(interpolate, 3), inputs=[13, 11, 10], output=13),
        Instruction(Function(torch.add, 2), inputs=[13, 8], output=15),
        Instruction(Function(clip, 2), inputs=[6, 15], output=9),
        Instruction(Function(torch.maximum, 2), inputs=[15, 8], output=15),
        Instruction(Function(torch.sqrt, 1), inputs=[15], output=17),
        Instruction(Function(torch.tan, 1), inputs=[7], output=14),
        Instruction(Function(torch.div, 2), inputs=[14, 17], output=19),
        Instruction(Function(torch.maximum, 2), inputs=[9, 17], output=13),
        Instruction(Function(torch.mul, 2), inputs=[19, 1], output=18),
    ]
)

benchmark = nosbench.create("pfn", path="mem", device="cuda")

loss, run = benchmark.query(Incumbent, 19, return_run=True)
training_losses = list(map(np.mean, run.results[0].training_losses))
plt.plot(training_losses[:20], label="Incumbent", color="orange")

loss, run = benchmark.query(AdamW, 19, return_run=True)
training_losses = list(map(np.mean, run.results[0].training_losses))
plt.plot(training_losses[:20], label="AdamW", color="red")

loss, run = benchmark.query(Adadelta, 19, return_run=True)
training_losses = list(map(np.mean, run.results[0].training_losses))
plt.plot(training_losses[:20], label="Adadelta", color="purple")

plt.ylim(-1.0, 2.0)
plt.title("Meta-training Task Learning Curves")
plt.legend()
plt.savefig("learning_curve.png", bbox_inches="tight")
