import copy

from zero import Pointer, Instruction, Program, MemoryType
from zero import create_optimizer
from regularized_evolution import RegularizedEvolution

import torch
import numpy as np


# TODO: Combine ops wiht namedtuple(operation, "func, type")
# TODO: Specify which position of the memory to use as the result
# TODO: Add scalar operations to step instructions
# TODO: Start from a valid optimizer (SGD) and mutate it to create the
# the population
# TODO: Add more operations
# TODO: Refactor
# TODO: A way to cache the optimizers and the results
# TODO: Recognize identical optimizers
# TODO: Parallelization
# TODO: Less memory (lazy initialization)
# TODO: Better tasks CIFAR10
# TODO: Speed up techniques

# TODO: Cache
# TODO: Recognize
# TODO: Memory Usage
# TODO: Used memory locations


unary_ops = [torch.sqrt]
binary_ops = [torch.div, torch.mul, torch.sub, torch.add]
MAX_MEMORY = 10


def random_setup_instruction(rng, memory_size, ro_partition):
    op = binary_ops[rng.randint(0, len(binary_ops))]
    in1 = rng.randint(0, memory_size)
    in2 = rng.randint(0, memory_size)
    out = rng.randint(ro_partition, memory_size)
    instruction = Instruction(
            op,
            Pointer(MemoryType.SCALAR, in1),
            Pointer(MemoryType.SCALAR, in2),
            Pointer(MemoryType.SCALAR, out))
    return instruction


def random_step_instruction(rng, memory_size, ro_partition):
    in1 = rng.randint(0, memory_size)
    op = binary_ops[rng.randint(0, len(binary_ops))]
    memory_type = rng.randint(0, len(MemoryType))
    in2 = rng.randint(0, memory_size)
    out = rng.randint(ro_partition, memory_size)
    instruction = Instruction(
            op,
            Pointer(memory_type, in1), # SCALAR, VECTOR
            Pointer(MemoryType.VECTOR, in2),
            Pointer(MemoryType.VECTOR, out))
    return instruction


def add_instruction_mutation(program, rng):
    program = copy.deepcopy(program)
    if rng.randint(0, 2): # Setup
        instruction = random_setup_instruction(rng, MAX_MEMORY, 1)
        pos = rng.randint(0, len(program.setup) + 1)
        program.setup.insert(pos, instruction)
    else: # Step
        instruction = random_step_instruction(rng, MAX_MEMORY, 1)
        pos = rng.randint(0, len(program.step) + 1)
        program.step.insert(pos, instruction)
    return program


def remove_instruction_mutation(program, rng):
    program = copy.deepcopy(program)
    if rng.randint(0, 2): # Setup
        if len(program.setup):
            pos = rng.randint(0, len(program.setup))
            program.setup.pop(pos)
    else: # Step
        if len(program.step):
            pos = rng.randint(0, len(program.step))
            program.step.pop(pos)
    return program


class NOS(RegularizedEvolution):
    @staticmethod
    def evaluate_element(element, **kwargs):
        torch.manual_seed(123)
        model = torch.nn.Linear(2, 1)
        optimizer_class = create_optimizer(element)
        optim = optimizer_class(model.parameters(), lr=0.01)
        initial_loss = torch.nn.functional.mse_loss(
            model(torch.tensor([0.25, 0.25])), torch.tensor([10.0])
        ).item()
        for _ in range(1000):
            output = model(torch.tensor([0.25, -0.25]))
            loss = torch.nn.functional.mse_loss(output, torch.tensor([10.0]))
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss = loss.item()
        return -loss

    @staticmethod
    def random_element(rng, **kwargs):
        program_setup = []
        program_step = []
        for _ in range(rng.randint(0, 10)):
            instruction = random_setup_instruction(rng, MAX_MEMORY, 1)
            program_setup.append(instruction)

        for _ in range(rng.randint(0, 10)):
            instruction = random_step_instruction(rng, MAX_MEMORY, 2)
            program_step.append(instruction)
        return Program(program_setup, program_step)

    @staticmethod
    def mutate_element(element, rng, **kwargs):
        mutation_type = rng.randint(0, 2)
        if mutation_type == 0:
            child = add_instruction_mutation(element, rng)
        elif mutation_type == 1:
            child = remove_instruction_mutation(element, rng)
        return child

re = NOS(100, 25, rng=np.random.RandomState(123))
for _ in range(10000):
    print(max(re.history, key=lambda x: x.fitness))
    re.step()

exit()


def change_setup_mutation(program):
    program = Program(*program)
    program_setup = []
    for _ in range(random.randint(0, 10)):
        instruction = random_setup_instruction(binary_ops, MAX_MEMORY, 2)
        program_setup.append(instruction)
    program = Program(program_setup, program.step)
    return program

def change_step_mutation(program):
    program = Program(*program)
    program_step = []
    for _ in range(random.randint(0, 10)):
        instruction = random_step_instruction(binary_ops, MAX_MEMORY, 2)
        program_step.append(instruction)
    program = Program(program.setup, program_step)
    return program

def modify_setup_argument_mutation(program):
    program = Program(*program)
    instruction = random.choice(program.setup)
    idx = random.randint(0, 2)
    if idx == 0:
        instruction.in1 = Pointer(instruction.in1.name, random.randint(0, MAX_MEMORY))
    elif idx == 1:
        instruction.in2 = Pointer(instruction.in2.name, random.randint(0, MAX_MEMORY))
    elif idx == 2:
        instruction.out = Pointer(instruction.out.name, random.randint(1, MAX_MEMORY))
    return program

def modify_step_argument_mutation(program):
    program = Program(*program)
    instruction = random.choice(program.step)
    idx = random.randint(0, 2)
    if idx == 0:
        instruction.in1 = Pointer(instruction.in1.name, random.randint(0, MAX_MEMORY))
    elif idx == 1:
        instruction.in2 = Pointer(instruction.in2.name, random.randint(0, MAX_MEMORY))
    elif idx == 2:
        instruction.out = Pointer(instruction.out.name, random.randint(2, MAX_MEMORY))
    return program



mutations = [add_setup_instruction_mutation, remove_setup_instruction_mutation,
        add_step_instruction_mutation, remove_step_instruction_mutation,
        change_setup_mutation, change_step_mutation,
        modify_setup_argument_mutation,
        modify_step_argument_mutation]



class RegularizedEvolution:
    def __init__(self, population_size, t, n):
        self.t = t
        self.n = n
        self.population_size = population_size
        self.population = []
        for _ in range(population_size):
            self.population.append(random_program(10, 10))

    def run(self, n):
        for i in range(n):
            for _ in range(self.n):
                self.population.pop(0)
            eval_idxs = [random.randint(0, self.population_size-self.n-1) for _ in range(self.t)]
            fitness = []
            for idx in eval_idxs:
                program = self.population[idx]
                fitness.append(evaluate(program))
            eval_idxs = np.array(eval_idxs)
            population = np.array(self.population, dtype=Program)
            n_best = population[eval_idxs[np.argsort(fitness)[:self.n]]]
            for best in n_best:
                clone = copy.deepcopy(best)
                mutate = random.choice(mutations)
                clone = mutate(clone)
                self.population.append(Program(*clone))
            print(np.min(fitness))



random.seed(123)
re = RegularizedEvolution(5, 5, 1)
print(re)
re.run(50)

