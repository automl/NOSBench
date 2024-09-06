NOSBench-101: Towards Reproducible Neural Optimizer Search
===========================================================

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

benchmark = nosbench.NOSBench()
# benchmark = nosbench.ToyBenchmark()
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

benchmark = nosbench.NOSBench()
print(benchmark.query(SGD, 10))
```

Check [nosbench/program.py](nosbench/program.py) for values in `READONLY_REGION`.

### Configuration Space

```python
import pprint
import nosbench

benchmark = nosbench.NOSBench()
cs = benchmark.configspace(seed=123)

for _ in range(5):
    config = cs.sample_configuration()
    program = benchmark.configuration_to_program(config)
    loss = benchmark.query(program, 10)
    pprint.pprint(program)
    print(loss)
```

### Pruning Optimizers
Resulting optimizers from a search usually contains a lot of redundant instructions. You can use [prune_program](https://github.com/automl/NOSBench/blob/5f7c859b39cce33d61a530b5008d4e04eeb28a1b/nosbench/utils.py#L21-L34) function to clean it.


## Citation

If you use `NOSBench-101` in your research, please cite our paper:

```bibtex
@inproceedings{
    karakasli2024nosbench,
    title={{NOSB}ench-101: Towards Reproducible Neural Optimizer Search},
    author={Goktug Karakasli and Steven Adriaensen and Frank Hutter},
    booktitle={AutoML Conference 2024 (Workshop Track)},
    year={2024},
    url={https://openreview.net/forum?id=5Lm2ghxMlp}
}
```
