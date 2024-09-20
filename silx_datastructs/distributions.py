from enum import Enum
from typing import Union, Optional
from itertools import groupby
from functools import reduce

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

    def __add__(self, val):
        if val.name != self.name:
            raise ValueError(
                f"Cannot add probabilities for different things {val.name}, {self.name}"
            )
        return SingleCountProbability(
            name=self.name,
            numerator=self.numerator + val.numerator,
            denominator=self.denominator + val.denominator,
        )

    def __mul__(self, val):
        if val.name != self.name:
            raise ValueError(
                f"Cannot add probabilities for different things {val.name}, {self.name}"
            )
        return SingleCountProbability(
            name=self.name,
            numerator=self.numerator * val.numerator,
            denominator=self.denominator * val.denominator,
        )


class CountDistribution(BaseModel):
    probabilities: list[SingleCountProbability]

    @property
    def denominator(self) -> int:
        return self.probabilities[0].denominator

    @denominator.setter
    def denominator(self, val: int) -> None:
        for i in range(len(self.probabilities)):
            self.probabilities[i].denominator = val

    def check(self) -> None:
        if not _all_equal(map(lambda x: x.denominator, self.probabilities)):
            raise ValueError("Invalid distribution: denominators not all equal")
        if not all(map(lambda x: x.is_valid(), self.probabilities)):
            raise ValueError("Invalid distribution: individual probabilities not valid")
        names = set(map(lambda x: x.name, self.probabilities))
        if len(names) != len(self.probabilities):
            raise ValueError("Invalid distribution: repeated elements")
        numerator_sum = sum(map(lambda p: p.numerator, self.probabilities))
        if numerator_sum != self.denominator:
            raise ValueError(
                f"Invalid distribution: numerator sum {numerator_sum}"
                "not equal to denominator {self.denominator}"
            )

    def name_lookup(self, name: str) -> SingleCountProbability:
        for p in self.probabilities:
            if p.name == name:
                return p
        raise ValueError(f"{name} not found in probability list")

    def generate(self) -> list[str]:
        self.check()
        output: list[str] = []
        for probability in self.probabilities:
            output.extend([probability.name] * probability.numerator)
        return output

    def __add__(self, val):
        val.check()
        self.check()

        # match names
        my_probs = {p.name: p for p in self.probabilities}
        input_probs = {p.name: p for p in val.probabilities}

        if my_probs.keys() != input_probs.keys():
            raise ValueError(
                f"cannot add different distributions"
                " {list(my_probs.keys())} {list(input_probs.keys())}"
            )

        new_probs: list[SingleCountProbability] = []
        for k, mp in my_probs.items():
            ip = input_probs[k]

            new_probs.append(mp + ip)

        return CountDistribution(probabilities=new_probs)

    def __mul__(self, val):
        val.check()
        self.check()

        # match names
        my_probs = {p.name: p for p in self.probabilities}
        input_probs = {p.name: p for p in val.probabilities}

        if my_probs.keys() != input_probs.keys():
            raise ValueError(
                f"cannot add different distributions"
                " {list(my_probs.keys())} {list(input_probs.keys())}"
            )

        new_probs: list[SingleCountProbability] = []
        for k, mp in my_probs.items():
            ip = input_probs[k]

            new_probs.append(mp * ip)

        return CountDistribution(probabilities=new_probs)


class NormalDistribution(BaseModel):
    mu: float
    sigma: float
    N: int

    def check(self) -> None:
        if self.mu < 0 or self.sigma < 0 or self.N < 1:
            raise ValueError(f"Invalid distribution {self}")

    def generate(self) -> list[float]:
        return list(np.random.normal(self.mu, self.sigma, self.N))

    def __add__(self, val):
        new_mu = self.mu + val.mu
        new_sigma = np.sqrt(self.sigma**2 + val.sigma**2)
        new_n = self.N + val.N
        return NormalDistribution(mu=new_mu, sigma=new_sigma, N=new_n)


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

    def __add__(self, val):
        # approximate
        sigma_self_2 = self.sigma**2
        sigma_val_2 = val.sigma**2

        en_1 = np.exp(2 * self.mu + sigma_self_2) * (np.exp(sigma_self_2) - 1)
        en_2 = np.exp(2 * val.mu + sigma_val_2) * (np.exp(sigma_val_2) - 1)
        ed_1 = np.exp(self.mu + (self.sigma**2 / 2))
        ed_2 = np.exp(val.mu + (val.sigma**2 / 2))

        sigma_2 = np.log(((en_1 + en_2) / (ed_1 + ed_2) ** 2) + 1)

        em_1 = np.exp(self.mu + (sigma_self_2 / 2))
        em_2 = np.exp(val.mu + (sigma_val_2 / 2))

        new_mu = np.log((em_1 + em_2) - 0.5 * sigma_2)
        new_sigma = np.sqrt(sigma_2)

        new_n = self.N + val.N
        return LogNormalDistribution(mu=new_mu, sigma=new_sigma, N=new_n)


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

    def __add__(self, _):
        raise SyntaxError("Unable to add NormalHazardRatio")


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

    def __add__(self, _):
        return MissingData(dummy=None)


class DoStatement(BaseModel):
    value: str | bool | int | float | CatRange


class DoDistribution(BaseModel):
    value: str | bool | int | float
    N: int

    def generate(self) -> list[str | bool | int | float]:
        return [self.value] * self.N

    def __add__(self, val):
        if val.value != self.value:
            raise ValueError("Can't add do distributions with different names")
        return DoDistribution(value=self.value, N=self.N + val.N)


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
    MissingData,
]


VARIABLE_T = Union[RAND_VAR_T, ConstVar]
