Neural Optimizer Search Benchmark
=================================

## Install

```bash
poetry install
```

## Usage

Check [examples](examples) folder for advanced usage.

### Query optimizers

```python
import nosbench
from nosbench.optimizers import AdamW

benchmark = nosbench.create("toy")
print(benchmark.query(AdamW, 10))
```

### Create your own optimizer

```python
import nosbench
from nosbench.program import Program, Instruction, Pointer
from nosbench.function import Function

SGD = Program(
    [
        # momentum = 1 - 0.1 = 0.9
        Instruction(Function(torch.sub, 2), [Pointer(3), Pointer(5)], Pointer(9)),
        # m = m * momentum
        Instruction(Function(torch.mul, 2), [Pointer(10), Pointer(9)], Pointer(10)),
        # m = gradient + m
        Instruction(Function(torch.add, 2), [Pointer(1), Pointer(10)], Pointer(10)),
        # update = m * 0.001
        Instruction(Function(torch.mul, 2), [Pointer(10), Pointer(6)], Pointer(11)),
    ]
)

benchmark = nosbench.create("pfn")
print(benchmark.query(SGD, 10))
```

Check [nosbench/program.py](nosbench/program.py) for values in `READONLY_REGION`.

### Configuration Space

Requires ConfigSpace
```bash
pip install ConfigSpace
```

```python
import pprint
import nosbench

benchmark = nosbench.create("toy")
cs = benchmark.configspace(seed=123)

for _ in range(5):
    config = cs.sample_configuration()
    program = benchmark.configuration_to_program(config)
    loss = benchmark.query(program, 10)
    pprint.pprint(program)
    print(loss)
```
