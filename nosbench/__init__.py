from collections import namedtuple
import importlib

from nosbench.program import READONLY_REGION
from nosbench.nosbench import NOSMLPBench, ToyBenchmark, NOSBench

registry = {}


Spec = namedtuple("Spec", "identifier, entry_point")


def register(name, identifier, entry_point):
    registry[name] = Spec(identifier, entry_point)


def create(name, path="cache", **kwargs):
    spec = registry[name]
    if callable(spec.entry_point):
        return spec.entry_point.from_identifier(spec.identifier, path=path, **kwargs)
    else:
        module_name, benchmark = name.split(":")
        module = importlib.import_module(module_name)
        cls = getattr(module, benchmark)
        return cls.from_identifier(spec.identifier, path=path, **kwargs)


register("toy", "toy", ToyBenchmark)
register("mlp-1", "2n312o102s606s3e2903n2723c2543h2363n2183o2001e3b128", NOSMLPBench)
register("mlp-2", "2n312o102s213h3e2013n2013c2001o3n128", NOSMLPBench)
register("mlp-3", "2n312o102s503h3e5293n3653c2001o3n128", NOSMLPBench)
register("mlp-4", "2n312o102s213h3e2783n2393c2001o3n128", NOSMLPBench)
register("pfn", "pfn", NOSBench)


__all__ = [
    "NOSBench",
    "NOSMLPBench",
    "Toybenchmark",
    "READONLY_REGION",
    "register",
    "create",
]
