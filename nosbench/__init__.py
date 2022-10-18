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
        return spec.entry_point.from_identifier(spec.identifier, **kwargs)
    else:
        module_name, benchmark = name.split(":")
        module = importlib.import_module(module_name)
        cls = getattr(module, benchmark)
        return cls.from_identifier(spec.identifier, **kwargs)


register("toy", "toy", ToyBenchmark)
register("mlp-1", "2n312o102s601b06s3e2903n2723c2543h2363n2183o2001e3b128", NOSBench)
register("mlp-2", "2n312o102s211b03h3e2013n2013c2001o3n128", NOSBench)
register("mlp-3", "2n312o102s501b03h3e5293n3653c2001o3n128", NOSBench)
register("mlp-4", "2n312o102s211b03h3e2783n2393c2001o3n128", NOSBench)


__all__ = ["NOSBench", "Toybenchmark", "READONLY_REGION", "register", "create"]
