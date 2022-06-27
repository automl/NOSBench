import abc
from functools import partial
from collections import namedtuple

import numpy as np


_Element = namedtuple("_Element", "cls fitness")


class RegularizedEvolution(abc.ABC):
    def __init__(
        self, population_size, tournament_size, rng=np.random.RandomState(), **kwargs
    ):
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.rng = rng
        self._evaluate_element = partial(self.evaluate_element, rng=rng, **kwargs)
        self._mutate_element = partial(self.mutate_element, rng=rng, **kwargs)
        self._random_element = partial(self.random_element, rng=rng, **kwargs)

        self.population = []
        self.history = []
        for _ in range(self.population_size):
            element = self._random_element()
            element = _Element(element, self._evaluate_element(element))
            self.population.append(element)
            self.history.append(element)

    @staticmethod
    @abc.abstractmethod
    def evaluate_element(element, rng, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def mutate_element(element, rng, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def random_element(rng, **kwargs):
        pass

    def step(self):
        idxs = self.rng.randint(0, self.population_size, self.tournament_size)
        samples = [self.population[idx] for idx in idxs]
        parent = max(samples, key=lambda x: x.fitness)
        child = self._mutate_element(parent.cls)
        child = _Element(child, self._evaluate_element(child))
        self.population.append(child)
        self.history.append(child)
        self.population.pop(0)


if __name__ == "__main__":
    Parameters = namedtuple("Parameters", "x y")

    class Ackley(RegularizedEvolution):
        @staticmethod
        def evaluate_element(element, **kwargs):
            first_term = -20 * np.exp(
                -0.2 * np.sqrt(0.5 * (element.x**2 + element.y**2))
            )
            second_term = (
                -np.exp(
                    0.5
                    * (np.cos(2 * np.pi * element.x) + np.cos(2 * np.pi * element.y))
                )
                + np.e
                + 20
            )
            return -(second_term + first_term)

        @staticmethod
        def mutate_element(element, rng, sigma, **kwargs):
            x = element.x
            y = element.y
            if rng.randint(0, 2) == 0:
                element = Parameters(x + rng.normal(scale=sigma), y)
            else:
                element = Parameters(x, y + rng.normal(scale=sigma))
            return element

        @staticmethod
        def random_element(rng, sigma, **kwargs):
            return Parameters(rng.normal(scale=sigma), rng.normal(scale=sigma))

    re = Ackley(100, 25, rng=np.random.RandomState(123), sigma=0.01)
    for _ in range(10000):
        print(max(re.history, key=lambda x: x.fitness))
        re.step()
