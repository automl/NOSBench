import pprint

from smac import BlackBoxFacade, Scenario
from smac.initial_design import RandomInitialDesign
import numpy as np

import nosbench


# Requires SMAC >= 2.0.0

benchmark = nosbench.create("toy")

scenario = Scenario(
    configspace=benchmark.configspace(seed=123),
    n_trials=200,
    output_directory="smac_results",
    deterministic=True,
)


def runner(config, seed):
    program = benchmark.configuration_to_program(config)
    loss = benchmark.query(program, 10)
    if np.isnan(loss) or np.isinf(loss):
        return 2147483648
    return loss


initial_design = RandomInitialDesign(scenario, n_configs=10)
smac = BlackBoxFacade(scenario, runner, overwrite=True, initial_design=initial_design)
config = smac.optimize()
pprint.pprint(benchmark.configuration_to_program(config))
