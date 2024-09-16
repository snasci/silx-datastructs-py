from typing import Protocol, Union, Optional
from itertools import groupby

import numpy.random as rand
from pydantic import BaseModel

from .dag import CatRange


def _all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


class DistributionProtocol(Protocol):
    def generate(self) -> list: ...


class SingleCountProbability(BaseModel):
    name: str
    numerator: int
    denominator: int

    def is_valid(self) -> bool:
        return self.numerator <= self.denominator

    def p(self) -> float:
        return self.numerator / self.denominator


class CountDistribution(BaseModel):
    probabilities: list[SingleCountProbability]

    def is_valid(self) -> bool:
        if not _all_equal(map(lambda x: x.denominator, self.probabilities)):
            return False
        if not all(map(lambda x: x.is_valid(), self.probabilities)):
            return False
        names = set(map(lambda x: x.name, self.probabilities))
        if len(names) != len(self.probabilities):
            return False
        return True

    def name_lookup(self, name: str) -> SingleCountProbability:
        for p in self.probabilities:
            if p.name == name:
                return p
        raise ValueError(f"{name} not found in probability list")

    def generate(self) -> list[str]:
        if not self.is_valid():
            raise ValueError(f"Invalid CountDistribution {self}")
        output: list[str] = []
        for probability in self.probabilities:
            output.extend([probability.name] * probability.numerator)
        return output


class NormalDistribution(BaseModel):
    mu: float
    sigma: float
    N: int

    def is_valid(self) -> bool:
        if self.mu < 0 or self.sigma < 0 or self.N < 1:
            return False
        return True

    def generate(self) -> list[float]:
        return list(rand.normal(self.mu, self.sigma, self.N))


class LogNormalDistribution(BaseModel):
    mu: float
    sigma: float
    N: int

    def is_valid(self) -> bool:
        if self.mu < 0 or self.sigma < 0 or self.N < 0:
            return False
        return True

    def generate(self) -> list[float]:
        return list(rand.lognormal(self.mu, self.sigma, self.N))


# Deprecated
class CountProbability(BaseModel):
    n: int
    total: int

    def __str__(self) -> str:
        s = f"COUNT({self.n}/{self.total})"
        return s

    def p(self) -> float:
        return self.n / self.total


class CoxLogRank(BaseModel):
    hr: float
    sigma: float


class MissingData(BaseModel):
    dummy: Optional[str] = None


class DoStatement(BaseModel):
    realization: str | bool | int | float | CatRange


class ConditionCategory(BaseModel):
    realization: str | bool


# Deprecated
class ConstVar(BaseModel):
    value: str | int | float | CatRange


RAND_VAR_T = Union[
    CountProbability,
    NormalDistribution,
    LogNormalDistribution,
    MissingData,
    DoStatement,
    ConditionCategory,
    CoxLogRank,
]


VARIABLE_T = Union[RAND_VAR_T, ConstVar]
