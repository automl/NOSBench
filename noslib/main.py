from typing import Any, Union, Callable
from dataclasses import dataclass
from sympy import symbols, Integer, Number
from sympy import Symbol, simplify, lambdify
import sympy
import sympy.core.expr as sexpr
from functools import singledispatch
import torch
from sympy.parsing.sympy_parser import parse_expr


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


@get_formula.register
def _(expr: Number):
    return expr


g, g_square, g_cube, m_hat, v_hat, y_hat = operands = symbols(
    ",".join(
        [
            "g",
            "g_square",
            "g_cube",
            "m_hat",
            "v_hat",
            "y_hat",
        ]
    )
)
operands += (Integer(1), Integer(2))

unary_functions = [
    lambda x: x,
    lambda x: -x,
    lambda x: sympy.log(sympy.Abs(x)),
    lambda x: sympy.sqrt(sympy.Abs(x)),
    sympy.sign,
]
binary_functions = [
    sympy.Add,
    lambda x, y: x - y,
    sympy.Mul,
    lambda x, y: x / (y + 1e-8),
    lambda x, y: x**y,
]

modules = {
    "log": lambda x: torch.log(torch.tensor(x)),
    "sqrt": lambda x: torch.sqrt(torch.tensor(x)),
}


def sample_random_tree(depth, rng):
    args = set()

    def _sample_tree(current_depth=0):
        type_idx = rng.randint(0, 3) if current_depth < depth else 0
        if type_idx == 0:
            symbol = operands[rng.randint(0, len(operands))]
            if isinstance(symbol, Symbol):
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


def sample_random_tree(max_depth, rng):
    args = set()
    depth = rng.randint(0, max_depth)

    def _sample_tree(current_depth=0):
        if current_depth == depth:
            s = operands[rng.randint(0, len(operands))]
            if isinstance(s, Symbol):
                args.add(s)
            return s
        s1 = _sample_tree(current_depth + 1)
        s2 = _sample_tree(current_depth + 1)
        u1 = unary_functions[rng.randint(0, len(unary_functions))]
        u2 = unary_functions[rng.randint(0, len(unary_functions))]
        op = binary_functions[rng.randint(0, len(binary_functions))]
        return BinaryOp(op, UnaryOp(u1, s1), UnaryOp(u2, s2))

    return _sample_tree(), args


def get_optimizer(formula, args):
    class Optimizer(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, **kwargs):
            defaults = dict(lr=lr, **kwargs)
            self.update_func = lambdify(list(args), formula, modules=modules)
            super(Optimizer, self).__init__(params, defaults)

        @torch.no_grad()
        def step(self):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        state = self.state[p]
                        if len(state) == 0:
                            state["step"] = torch.tensor(0.0)
                            if m_hat in args:
                                # Exponential moving average of gradient values
                                state["exp_avg"] = torch.zeros_like(
                                    p, memory_format=torch.preserve_format
                                )
                            if v_hat in args:
                                # Exponential moving average of squared gradient values
                                state["exp_avg_sq"] = torch.zeros_like(
                                    p, memory_format=torch.preserve_format
                                )
                            if y_hat in args:
                                # Exponential moving average of cubed gradient values
                                state["exp_avg_cube"] = torch.zeros_like(
                                    p, memory_format=torch.preserve_format
                                )

                        state["step"] += 1
                        step = state["step"].item()

                        # if weight_decay != 0:
                        #     grad = grad.add(param, alpha=weight_decay)

                        kwargs = {}
                        if g in args:
                            kwargs["g"] = p.grad

                        if m_hat in args:
                            bias_correction1 = 1 - self.defaults["betas"][0] ** step
                            state["exp_avg"].mul_(self.defaults["betas"][0]).add_(
                                p.grad, alpha=1 - self.defaults["betas"][0]
                            )
                            exp_avg_hat = state["exp_avg"] / bias_correction1
                            kwargs["m_hat"] = exp_avg_hat

                        if v_hat in args or g_square in args:
                            grad_sq = p.grad**2
                            if g_square in args:
                                kwargs["g_square"] = grad_sq

                        if v_hat in args:
                            bias_correction2 = 1 - self.defaults["betas"][1] ** step
                            state["exp_avg_sq"].mul_(self.defaults["betas"][1]).add_(
                                grad_sq, alpha=1 - self.defaults["betas"][1]
                            )
                            exp_avg_sq_hat = state["exp_avg_sq"] / bias_correction2
                            kwargs["v_hat"] = exp_avg_sq_hat

                        if y_hat in args or g_cube in args:
                            grad_cube = p.grad**3
                            if g_cube in args:
                                kwargs["g_cube"] = grad_cube

                        if y_hat in args:
                            bias_correction3 = 1 - self.defaults["betas"][2] ** step
                            state["exp_avg_cube"].mul_(self.defaults["betas"][1]).add_(
                                grad_cube, alpha=1 - self.defaults["betas"][1]
                            )
                            exp_avg_cube_hat = state["exp_avg_cube"] / bias_correction3
                            kwargs["y_hat"] = exp_avg_cube_hat

                        d_p = -self.update_func(**kwargs)
                        p.add_(d_p, alpha=self.defaults["lr"])

    return Optimizer


def test_train(optimizer_class, steps):
    torch.manual_seed(123)
    try:
        model = torch.nn.Linear(1, 1)
        optim = optimizer_class(model.parameters(), lr=0.01, betas=(0.9, 0.999, 0.999))
        initial_loss = torch.nn.functional.mse_loss(
            model(torch.tensor([1.0])), torch.tensor([1.0])
        ).item()
        for _ in range(steps):
            output = model(torch.tensor([1.0]))
            loss = torch.nn.functional.mse_loss(output, torch.tensor([1.0]))
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss = loss.item()
    except:
        initial_loss = loss = np.nan
    return initial_loss, loss


import numpy as np

torch.manual_seed(123)
rng = np.random.RandomState(123)

# sgd test
sgd = get_optimizer(g, [g])
print(f"SGD: {test_train(sgd, 100)}")

# adam test
adam = get_optimizer(m_hat / (v_hat + 1e-8), [m_hat, v_hat])
print(f"Adam: {test_train(adam, 100)}")

# random optimizer test
results = []
for _ in range(100):
    tree, args = sample_random_tree(5, rng)
    formula = get_formula(tree)
    optimizer_class = get_optimizer(formula, args)
    initial_loss, loss = test_train(optimizer_class, 100)
    results.append((formula, loss))
    print(f"Formula: {formula}, Initial loss: {initial_loss} Final loss: {loss}")

results = np.array(results)
idx = np.nanargmin(results[:, 1])
print(f"Best:\n{results[idx]}")
