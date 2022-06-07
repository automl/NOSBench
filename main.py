from typing import Any, Union, Callable
from dataclasses import dataclass
from sympy import symbols
from sympy import Symbol, simplify, lambdify
import sympy
import sympy.core.expr as sexpr
from functools import singledispatch
import torch


Expr = Union['UnaryOp', 'BinaryOp', Symbol]


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

g, g_square, g_cube, m_hat, v_hat, y_hat, test = operands = symbols('g, g_square, g_cube, m_hat, v_hat, y_hat, test')
unary_functions = [lambda x: x, sympy.log]
binary_functions = [sympy.Add, lambda x, y: x - y, sympy.Mul]


def sample_random_tree(depth, rng):
    def _sample_tree(current_depth=0):
        type_idx = rng.randint(0, 3) if current_depth < depth else 0
        if type_idx == 0:
            return operands[rng.randint(0, len(operands))]
        elif type_idx == 1:
            op = unary_functions[rng.randint(0, len(unary_functions))]
            expr = _sample_tree(current_depth+1)
            return UnaryOp(op, expr)
        elif type_idx == 2:
            op = binary_functions[rng.randint(0, len(binary_functions))]
            left_expr = _sample_tree(current_depth+1)
            right_expr = _sample_tree(current_depth+1)
            return BinaryOp(op, left_expr, right_expr)
        else:
            raise NotImplementedError
    return _sample_tree()


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
                    _generate_gen(current_depth+1, beginning+[type_idx, i])
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


# lambdify(modules={'log': torch.log})

import numpy as np
rng = np.random.RandomState(123)
exps = []
for _ in range(10):
    exps.append(get_formula(sample_random_tree(5, rng)))
func = lambdify([g_square, v_hat, y_hat], exps[1], modules={'log': torch.log})
print(func(torch.ones(3), torch.ones(3), torch.ones(3)))

import sys
sys.setrecursionlimit(10000)
import pprint
print(len(sample_everything(6)))
exit()
import numpy as np

exit()

x = Symbol('x')
print(x.as_expr() == 'x')
print(x == 'x')
print(dir(x))
exit()
y = Symbol('y')

a = BinaryOp(lambda x, y: x + y, UnaryOp(lambda x: -x, x), y)
expr1 = simplify(get_formula(a))

a = BinaryOp(lambda x, y: y + x, UnaryOp(lambda y: y, y), UnaryOp(lambda x: -x, x))
expr2 = simplify(get_formula(a))
print(expr1, expr2)

print(expr1 == expr2)

print(lambdify([x, y], expr1)(10, 20))
