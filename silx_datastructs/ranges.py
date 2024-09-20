from pydantic import BaseModel
import numpy as np


class LeftRangeFragment(BaseModel):
    number: float
    inclusive: bool

    def __str__(self):
        bracket = "[" if self.inclusive else "("
        return f"{bracket}{self.number}"


class RightRangeFragment(BaseModel):
    number: float
    inclusive: bool

    def __str__(self):
        bracket = "]" if self.inclusive else ")"
        return f"{self.number}{bracket}"


TFragment = LeftRangeFragment | RightRangeFragment


class RangeStatement(BaseModel):
    left: LeftRangeFragment | None
    right: RightRangeFragment | None

    def __str__(self) -> str:
        left_str = "(_" if self.left is None else str(self.left)
        right_str = "_)" if self.right is None else str(self.right)
        return f"{left_str}, {right_str}"

    def distance(self, query_value: float) -> float:
        left_val = -np.inf
        left_inc = False
        if self.left is not None:
            left_val = self.left.number
            left_inc = self.left.inclusive

        right_val = np.inf
        right_inc = False
        if self.right is not None:
            right_val = self.right.number
            right_inc = self.right.inclusive

        inside = False
        if left_inc and right_inc:
            inside = left_val <= query_value <= right_val
        elif not left_inc and right_inc:
            inside = left_val < query_value <= right_val
        elif left_inc and not right_inc:
            inside = left_val <= query_value < right_val
        elif not left_inc and not right_inc:
            inside = left_val < query_value < right_val
        if inside:
            return 0

        # Check if above
        if right_inc:
            above = right_val <= query_value
            if above:
                diff = query_value - right_val
                if diff == 0:
                    diff += 1e6
                return diff
        else:
            above = right_val < query_value
            if above:
                diff = query_value - right_val
                return diff

        # check if below
        if left_inc:
            below = left_val >= query_value
            if below:
                diff = left_val - query_value
                if diff == 0:
                    diff += 1e6
                return diff
        else:
            below = left_val > query_value
            if below:
                diff = left_val - query_value
                return diff
        raise ValueError(f"Something went wrong comparing {query_value} to {str(self)}")
