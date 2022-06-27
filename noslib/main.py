exit()
unary_ops = [torch.sqrt]
binary_ops = [torch.div, torch.mul, torch.sub, torch.add]

MAX_MEMORY = 10

def random_setup_instruction(binary_ops, memory_size, read_only_partition):
    op = random.choice(binary_ops)
    in1 = random.randint(0, memory_size)
    in2 = random.randint(0, memory_size)
    out = random.randint(read_only_partition, memory_size)
    instruction = Instruction(
            op,
            Pointer(MemoryType.SCALAR, in1),
            Pointer(MemoryType.SCALAR, in2),
            Pointer(MemoryType.SCALAR, out))
    return instruction


def random_step_instruction(binary_ops, memory_size, read_only_partition):
    in1 = random.randint(0, memory_size)
    a = random.randint(0, 1)
    if  a == 0:
        op = random.choice(unary_ops)
    else:
        op = random.choice(binary_ops)
        memory_type = random.randint(0, len(MemoryType)-1)
        in2 = random.randint(0, memory_size)
    out = random.randint(read_only_partition, memory_size)

    if a == 1:
        instruction = Instruction(
                op,
                Pointer(memory_type, in1), # SCALAR, VECTOR
                Pointer(MemoryType.VECTOR, in2),
                Pointer(MemoryType.VECTOR, out))
    else:
        instruction = Instruction(
                op,
                Pointer(MemoryType.VECTOR, in1), # SCALAR, VECTOR
                None,
                Pointer(MemoryType.VECTOR, out))
    return instruction


def random_program(max_n_setup, max_n_step):
    program_setup = []
    program_step = []
    for _ in range(random.randint(0, max_n_setup)):
        instruction = random_setup_instruction(binary_ops, MAX_MEMORY, 1)
        program_setup.append(instruction)

    for _ in range(random.randint(0, max_n_step)):
        instruction = random_step_instruction(binary_ops, MAX_MEMORY, 2)
        program_step.append(instruction)

    return Program(program_setup, program_step)



def evaluate(program):
    torch.manual_seed(123)
    model = torch.nn.Linear(2, 1)
    optimizer_class = create_optimizer(program)
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
    return loss


def add_setup_instruction_mutation(program):
    program = Program(*program)
    instruction = random_setup_instruction(binary_ops, MAX_MEMORY, 1)
    pos = random.randint(0, len(program.setup))
    program.setup.insert(pos, instruction)
    return program

def remove_setup_instruction_mutation(program):
    program = Program(*program)
    if len(program.setup):
        pos = random.randint(0, len(program.setup)-1)
        program.setup.pop(pos)
    return program

def add_step_instruction_mutation(program):
    program = Program(*program)
    instruction = random_step_instruction(binary_ops, MAX_MEMORY, 1)
    pos = random.randint(0, len(program.step))
    program.step.insert(pos, instruction)
    return program

def remove_step_instruction_mutation(program):
    program = Program(*program)
    if len(program.step):
        pos = random.randint(0, len(program.step)-1)
        program.step.pop(pos)
    return program

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

