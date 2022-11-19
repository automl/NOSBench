import pprint
from collections import namedtuple
import json
from pathlib import Path
import pickle
import time
import argparse

import nosbench


_Element = namedtuple("_Element", "cls fitness")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--benchmark_name", type=str, default="toy")
    parser.add_argument("--benchmark_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--evaluations", type=int, default=100000)
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()

    benchmark = nosbench.create(args.benchmark_name)

    cs = benchmark.configspace(seed=args.seed)
    history = []

    timestr = time.strftime("%Y-%m-%d")
    settings = {"search_algorithm": "RS", "args": vars(args)}
    dump = json.dumps(settings, sort_keys=True)
    path = Path(args.output_path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"{timestr}-{hash(dump)}.json", "w") as f:
        f.write(dump)

    for i in range(args.evaluations):
        config = cs.sample_configuration()
        program = benchmark.configuration_to_program(config)
        fitness = -benchmark.query(program, args.benchmark_epochs)
        history.append(_Element(program, fitness))
        x = max(history, key=lambda x: x.fitness)
        print(f"Evaluations: {i+1}, Fitness: {x.fitness}")

        if ((i % args.save_every) == 0 and i > 0) or (i >= args.evaluations - 1):
            with open(path / f"{timestr}-{hash(dump)}.pickle", "wb") as f:
                pickle.dump(history, f)

    x = max(history, key=lambda x: x.fitness)
    print("Incumbent optimizer:")
    pprint.pprint(x.cls)
