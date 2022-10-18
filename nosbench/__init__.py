from collections import namedtuple
import importlib

from nosbench.program import READONLY_REGION
from nosbench.nosbench import NOSBench, ToyBenchmark

registry = {}


Spec = namedtuple("Spec", "identifier, entry_point")


def register(name, identifier, entry_point):
    registry[name] = Spec(identifier, entry_point)


def create(name, path="cache", **kwargs):
    spec = registry[name]
    if callable(spec.entry_point):
        return spec.entry_point.from_identifier(spec.identifier)
    else:
        module_name, benchmark = name.split(":")
        module = importlib.import_module(module_name)
        cls = getattr(module, benchmark)
        return cls.from_identifier(spec.identifier)


register("toy", "toy", ToyBenchmark)
register("nosbench", "2n312o102s601b06s3e2903n2723c2543h2363n2183o2001e3b120", NOSBench)


__all__ = ["NOSBench", "Toybenchmark", "READONLY_REGION", "register", "create"]
