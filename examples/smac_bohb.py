import pprint

from smac import MultiFidelityFacade, Scenario
from smac.initial_design import RandomInitialDesign
import numpy as np

import nosbench


# Requires SMAC >= 2.0.0

benchmark = nosbench.create("toy")

scenario = Scenario(
    configspace=benchmark.configspace(seed=123),
    n_trials=200,
    min_budget=1,
    max_budget=10,
    output_directory="smac_results",
    deterministic=True,
)


def runner(config, budget, seed):
    program = benchmark.configuration_to_program(config)
    loss = benchmark.query(program, int(budget))
    if np.isnan(loss) or np.isinf(loss):
        return 2147483648
    return loss


initial_design = RandomInitialDesign(scenario, n_configs=10)
smac = MultiFidelityFacade(
    scenario, runner, overwrite=True, initial_design=initial_design
)
config = smac.optimize()
pprint.pprint(benchmark.configuration_to_program(config))
