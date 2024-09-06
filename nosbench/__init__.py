from collections import namedtuple
import importlib

from nosbench.program import READONLY_REGION
from nosbench.nosbench import NOSMLPBench, ToyBenchmark, NOSBench

__all__ = [
    "NOSBench",
    "NOSMLPBench",
    "Toybenchmark",
    "READONLY_REGION",
]
