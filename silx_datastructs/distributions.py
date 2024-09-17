from enum import Enum
from typing import Union, Optional
from itertools import groupby

import numpy as np
from pydantic import BaseModel

from .dag import CatRange


def _all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


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
        return list(np.random.normal(self.mu, self.sigma, self.N))


class LogNormalDistribution(BaseModel):
    mu: float
    sigma: float
    N: int

    def is_valid(self) -> bool:
        if self.mu < 0 or self.sigma < 0 or self.N < 0:
            return False
        return True

    def generate(self) -> list[float]:
        return list(np.random.lognormal(self.mu, self.sigma, self.N))


class RiskType(Enum):
    LINEAR = 0
    SQUARE = 1
    GAUSSIAN = 2


def _risk_function(
    x: np.ndarray,
    w: np.ndarray,
    risk_parameter: float,
    risk_type: RiskType = RiskType.GAUSSIAN,
) -> np.ndarray:
    risk = np.dot(x, w)
    match risk_type:
        case RiskType.LINEAR:
            return risk.reshape(-1, 1)
        case RiskType.SQUARE:
            risk = np.square(risk * risk_parameter)
            return risk.reshape(-1, 1)
        case RiskType.GAUSSIAN:
            risk = np.square(risk)
            risk = np.exp(-risk * risk_parameter)
            return risk.reshape(-1, 1)


class NormalHazardRatio(BaseModel):
    hazard_ratio: float
    sigma: float
    N: int

    def generate(self) -> list[float]:
        return list(np.random.normal(self.hazard_ratio, self.sigma, self.N))


# Adapted from pysurvival lib
class ExponentialSurvivalDistribution(BaseModel):
    hazard_rate: float
    lower_ci: float
    upper_ci: float

    def generate(self) -> list[float]:
        beta_0 = np.log(self.lower_ci)
        beta_1 = np.log(self.upper_ci)
        raise NotImplementedError("Not ready yet")


# Deprecated
class CountProbability(BaseModel):
    n: int
    total: int

    def __str__(self) -> str:
        s = f"COUNT({self.n}/{self.total})"
        return s

    def p(self) -> float:
        return self.n / self.total


# deprecated
class CoxLogRank(BaseModel):
    hr: float
    sigma: float
    N: int


class MissingData(BaseModel):
    dummy: Optional[str] = None


class DoStatement(BaseModel):
    value: str | bool | int | float | CatRange


class DoDistribution(BaseModel):
    value: str | bool | int | float
    N: int

    def generate(self) -> list[str | bool | int | float]:
        return [self.value] * self.N


class ConditionCategory(BaseModel):
    realization: str | bool


# Deprecated
class ConstVar(BaseModel):
    value: str | int | float | CatRange


# Deprecate
RAND_VAR_T = Union[
    CountProbability,
    NormalDistribution,
    LogNormalDistribution,
    MissingData,
    DoStatement,
    ConditionCategory,
    NormalDistribution,
]

DISTRIBUTION_T = Union[
    CountDistribution,
    NormalDistribution,
    LogNormalDistribution,
    DoDistribution,
    NormalHazardRatio,
]


VARIABLE_T = Union[RAND_VAR_T, ConstVar]
