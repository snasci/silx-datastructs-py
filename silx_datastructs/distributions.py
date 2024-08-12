from typing import Union, Optional

from pydantic import BaseModel
from .dag import CatRange


class CountProbability(BaseModel):
    n: int
    total: int

    def __str__(self) -> str:
        s = f"COUNT({self.n}/{self.total})"
        return s

    def p(self) -> float:
        return self.n / self.total


class NormalDistribution(BaseModel):
    mu: float
    sigma: float

    def __str__(self) -> str:
        return f"NORMAL(μ={self.mu}, σ={self.sigma})"


class LogNormalDistribution(BaseModel):
    mu: float
    sigma: float

    def __str__(self) -> str:
        return f"LOGNORMAL(μ={self.mu}, σ={self.sigma})"


class CoxLogRank(BaseModel):
    hr: float
    sigma: float


class MissingData(BaseModel):
    dummy: Optional[str] = None


class DoStatement(BaseModel):
    realization: str | bool | int | float | CatRange


class ConditionCategory(BaseModel):
    realization: str | bool


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
