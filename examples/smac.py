from smac.facade.smac_bb_facade import SMAC4BB
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.scenario.scenario import Scenario
import numpy as np

from nosbench import NOSBench

benchmark = NOSBench()

scenario = Scenario(
    {
        "run_obj": "quality",
        "runcount-limit": 1000,
        "cs": benchmark.configspace(seed=123),
        "abort_on_first_run_crash": False,
        "output_dir": "smac",
        "deterministic": True,
    }
)


def runner(config):
    program = benchmark.configuration_to_program(config)
    loss = benchmark.query(program, 10)
    if np.isnan(loss) or np.isinf(loss):
        return 2147483648
    return loss


smac = SMAC4BB(
    scenario=scenario, tae_runner=runner, rng=123, initial_design=RandomConfigurations
)
config = smac.optimize()
pprint.pprint(benchmark.configuration_to_program(config))
