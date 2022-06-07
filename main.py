from typing import Any, Union, Callable
from dataclasses import dataclass
from sympy import symbols
from sympy import Symbol, simplify, lambdify
import sympy
import sympy.core.expr as sexpr
from functools import singledispatch
import torch


Expr = Union["UnaryOp", "BinaryOp", Symbol]


@dataclass
class UnaryOp:
    operator: Callable[[sexpr.Expr], sexpr.Expr]
    expr: Expr


@dataclass
class BinaryOp:
    operator: Callable[[sexpr.Expr, sexpr.Expr], sexpr.Expr]
    left_expr: Expr
    right_expr: Expr


@singledispatch
def get_formula(expr):
    raise NotImplementedError


@get_formula.register
def _(expr: UnaryOp):
    return expr.operator(get_formula(expr.expr))


@get_formula.register
def _(expr: BinaryOp):
    left = get_formula(expr.left_expr)
    right = get_formula(expr.right_expr)
    return expr.operator(left, right)


@get_formula.register
def _(expr: Symbol):
    return expr


g, g_square, g_cube, m_hat, v_hat, y_hat = operands = symbols(
    "g, g_square, g_cube, m_hat, v_hat, y_hat"
)
# m_hat, v_hat, y_hat = operands = symbols("m_hat, v_hat, y_hat")
unary_functions = [lambda x: x, sympy.log]
binary_functions = [sympy.Add, lambda x, y: x - y, sympy.Mul]


def sample_random_tree(depth, rng):
    args = set()

    def _sample_tree(current_depth=0):
        type_idx = rng.randint(0, 3) if current_depth < depth else 0
        if type_idx == 0:
            symbol = operands[rng.randint(0, len(operands))]
            args.add(symbol)
            return symbol
        elif type_idx == 1:
            op = unary_functions[rng.randint(0, len(unary_functions))]
            expr = _sample_tree(current_depth + 1)
            return UnaryOp(op, expr)
        elif type_idx == 2:
            op = binary_functions[rng.randint(0, len(binary_functions))]
            left_expr = _sample_tree(current_depth + 1)
            right_expr = _sample_tree(current_depth + 1)
            return BinaryOp(op, left_expr, right_expr)
        else:
            raise NotImplementedError

    return _sample_tree(), args


def get_optimizer(formula, args, modules):
    class Optimizer(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, **kwargs):
            defaults = dict(lr=lr, **kwargs)
            self.update_func = lambdify(args, formula, modules=modules)
            super(Optimizer, self).__init__(params, defaults)

        @torch.no_grad()
        def step(self):
            for group in self.param_groups:
                params_with_grad = []
                grads = []
                calc_m_hat = m_hat in args
                calc_v_hat = v_hat in args
                calc_y_hat = y_hat in args
                calc_g_square = g_square in args
                calc_g_cube = g_cube in args
                calc_g = g in args

                exp_avgs = []
                exp_avg_sqs = []
                exp_avg_cubes = []
                state_steps = []

                for p in group["params"]:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        grads.append(p.grad)

                        state = self.state[p]
                        # Lazy state initialization
                        if len(state) == 0:
                            state["step"] = torch.tensor(0.0)
                            if calc_m_hat:
                                # Exponential moving average of gradient values
                                state["exp_avg"] = torch.zeros_like(
                                    p, memory_format=torch.preserve_format
                                )
                            if calc_v_hat:
                                # Exponential moving average of squared gradient values
                                state["exp_avg_sq"] = torch.zeros_like(
                                    p, memory_format=torch.preserve_format
                                )
                            if calc_y_hat:
                                # Exponential moving average of cubed gradient values
                                state["exp_avg_cube"] = torch.zeros_like(
                                    p, memory_format=torch.preserve_format
                                )

                        state["step"] += 1
                        step = state["step"].item()

                        # if weight_decay != 0:
                        #     grad = grad.add(param, alpha=weight_decay)

                        kwargs = {}
                        if calc_g:
                            kwargs["g"] = p.grad

                        if calc_m_hat:
                            bias_correction1 = 1 - self.defaults["betas"][0] ** step
                            state["exp_avg"].mul_(self.defaults["betas"][0]).add_(
                                p.grad, alpha=1 - self.defaults["betas"][0]
                            )
                            exp_avg_hat = state["exp_avg"] / bias_correction1
                            kwargs["m_hat"] = exp_avg_hat

                        if calc_v_hat or calc_g_square:
                            grad_sq = p.grad**2
                            if calc_g_square:
                                kwargs["g_square"] = grad_sq

                        if calc_v_hat:
                            bias_correction2 = 1 - self.defaults["betas"][1] ** step
                            state["exp_avg_sq"].mul_(self.defaults["betas"][1]).add_(
                                grad_sq, alpha=1 - self.defaults["betas"][1]
                            )
                            exp_avg_sq_hat = state["exp_avg_sq"] / bias_correction2
                            kwargs["v_hat"] = exp_avg_sq_hat

                        if calc_y_hat or calc_g_cube:
                            grad_cube = p.grad**3
                            if calc_g_cube:
                                kwargs["g_cube"] = grad_cube

                        if calc_y_hat:
                            bias_correction3 = 1 - self.defaults["betas"][2] ** step
                            state["exp_avg_cube"].mul_(self.defaults["betas"][1]).add_(
                                grad_cube, alpha=1 - self.defaults["betas"][1]
                            )
                            exp_avg_cube_hat = state["exp_avg_cube"] / bias_correction3
                            kwargs["y_hat"] = exp_avg_cube_hat

                        d_p = self.update_func(**kwargs)
                        p.add_(d_p, alpha=self.defaults["lr"])

    return Optimizer


import numpy as np

torch.manual_seed(123)
rng = np.random.RandomState(123)
exps = []


for _ in range(5):
    tree, args = sample_random_tree(5, rng)
    formula = get_formula(tree)
    optimizer_class = get_optimizer(formula, list(args), modules={"log": torch.log})
    model = torch.nn.Linear(10, 20)
    optim = optimizer_class(model.parameters(), betas=(0.9, 0.999, 0.999))
    output = model(torch.randn(10))
    loss = torch.nn.functional.mse_loss(output, torch.randn(20))
    optim.zero_grad()
    loss.backward()
    optim.step()
# get_optimizer(
exit()
# lambdify(modules={'log': torch.log})

func = lambdify([g_square, v_hat, y_hat], exps[1], modules={"log": torch.log})
print(func(torch.ones(3), torch.ones(3), torch.ones(3)))

import sys

sys.setrecursionlimit(10000)
import pprint

print(len(sample_everything(6)))
exit()
import numpy as np

exit()

x = Symbol("x")
print(x.as_expr() == "x")
print(x == "x")
print(dir(x))
exit()
y = Symbol("y")

a = BinaryOp(lambda x, y: x + y, UnaryOp(lambda x: -x, x), y)
expr1 = simplify(get_formula(a))

a = BinaryOp(lambda x, y: y + x, UnaryOp(lambda y: y, y), UnaryOp(lambda x: -x, x))
expr2 = simplify(get_formula(a))
print(expr1, expr2)

print(expr1 == expr2)

print(lambdify([x, y], expr1)(10, 20))


def sample_everything(depth):
    generator = []

    def _generate_gen(current_depth=0, beginning=[]):
        if current_depth >= depth:
            return
        for type_idx in range(3):
            if type_idx == 0:
                for i in range(len(operands)):
                    generator.extend(beginning)
                    generator.append(type_idx)
                    generator.append(i)
            elif type_idx == 1:
                for i in range(len(unary_functions)):
                    _generate_gen(current_depth + 1, beginning + [type_idx, i])
            # elif type_idx == 2:
            #     # THIS IS WRONG
            #     for i in range(len(binary_functions)):
            #         _generate_gen(current_depth+1, beginning+[type_idx, i])
            #         _generate_gen(current_depth+1)

    _generate_gen()
    gen = iter(generator)
    samples = []

    def _sample():
        type_idx = next(gen)
        if type_idx == 0:
            return operands[next(gen)]
        elif type_idx == 1:
            op = unary_functions[next(gen)]
            expr = _sample()
            return UnaryOp(op, expr)
        elif type_idx == 2:
            op = binary_functions[next(gen)]
            left_expr = _sample()
            right_expr = _sample()
            return BinaryOp(op, left_expr, right_expr)
        else:
            raise NotImplementedError

    try:
        while True:
            samples.append(_sample())
    except StopIteration:
        return samples
    assert False
