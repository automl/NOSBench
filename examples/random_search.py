import pprint
from collections import namedtuple
import json
from pathlib import Path
import pickle
import time
import argparse

import matplotlib.pyplot as plt

import nosbench


_Element = namedtuple("_Element", "cls fitness")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--cache_path", type=str, default="cache")
    parser.add_argument("--benchmark_name", type=str, default="toy")
    parser.add_argument("--benchmark_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--evaluations", type=int, default=100000)
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()

    benchmark = nosbench.create(args.benchmark_name, path=args.cache_path, device=args.device)

    cs = benchmark.configspace(seed=args.seed)
    history = []

    timestr = time.strftime("%Y-%m-%d")
    settings = {"search_algorithm": "RS", "args": vars(args)}
    dump = json.dumps(settings, sort_keys=True)
    path = Path(args.output_path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"{timestr}-{hash(dump)}.json", "w") as f:
        f.write(dump)

    import glob
    hits = []
    nans = []
    infs = []
    cache_sizes = []
    for i in range(args.evaluations):
        config = cs.sample_configuration()
        program = benchmark.configuration_to_program(config)
        fitness = -benchmark.query(program, args.benchmark_epochs)
        history.append(_Element(program, fitness))
        x = max(history, key=lambda x: x.fitness)
        # TODO: Why this never prints nan
        print(f"Evaluations: {i+1}, Fitness: {x.fitness}")
        hits.append(benchmark.stats.hits)
        nans.append(benchmark.stats.nans)
        infs.append(benchmark.stats.infs)
        print(f"Number of Queries: {benchmark.stats.n_queries}, Hits: {benchmark.stats.hits}, NaNs: {benchmark.stats.nans}, Infs: {benchmark.stats.infs}")

        cache_sizes.append(len(list((Path(args.cache_path) / args.benchmark_name).glob("*.run"))))

        if ((i % args.save_every) == 0 and i > 0) or (i >= args.evaluations - 1):
            with open(path / f"{timestr}-{hash(dump)}.pickle", "wb") as f:
                pickle.dump(history, f)

    plt.plot(cache_sizes, label="Cache Size", color="blue")
    plt.plot(hits, label="Hits", color="green")
    plt.plot(nans, label="NaNs", color="red")
    plt.plot(infs, label="Infs", color="orange")
    plt.xscale('log')
    plt.legend()
    plt.savefig('hit_miss_by_time_rs.png', bbox_inches='tight')
    
    print(f"Number of Queries: {benchmark.stats.n_queries}, Hits: {benchmark.stats.hits}, NaNs: {benchmark.stats.nans}, Infs: {benchmark.stats.infs}")
    x = max(history, key=lambda x: x.fitness)
    print("Incumbent optimizer:")
    pprint.pprint(x.cls)
