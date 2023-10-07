from abc import ABCMeta, abstractmethod
from typing import Type, Union
from backtester.SimulationData import SimulationData
import pandas as pd

class OutOfDataRangeError(IndexError):
    pass

class Expression(metaclass=ABCMeta):
    def __repr__(self) -> str: return str(self)

    @abstractmethod
    def evaluate(self) -> pd.DataFrame: ...

    @property
    def is_feature(self): raise NotImplementedError


class Feature(Expression):
    def __init__(self, feature: SimulationData) -> None:
        self._feature = feature

    def __str__(self) -> str: return '$' + self._feature.__name__().lower()

    @property
    def is_feature(self): return True

    def evaluate(self) -> pd.DataFrame:
        return self._feature.get_data()


class Constant(Expression):
    def __init__(self, value: float) -> None:
        self._value = value

    def __str__(self) -> str: return f'Constant({str(self._value)})'

    @property
    def is_feature(self): return False

    def evaluate(self) -> pd.DataFrame:
        # No need to be a dataframe due to broadcasting
        return self._value


class DeltaTime(Expression):
    # This is not something that should be in the final expression
    # It is only here for simplicity in the implementation of the tree builder
    def __init__(self, delta_time: int) -> None:
        # Validate that delta_time is a positive integer
        assert isinstance(delta_time, int) and delta_time > 0
        self._delta_time = delta_time

    def __str__(self) -> str: return str(self._delta_time)

    @property
    def is_feature(self): return False

    def evaluate(self) -> pd.DataFrame:
        # No need to be a dataframe due to broadcasting
        return self._delta_time


# Operator base classes

class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int: ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type['Operator']: ...


class UnaryOperator(Operator):
    def __init__(self, operand: Union[Expression, float]) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)

    @classmethod
    def n_args(cls) -> int: return 1

    @classmethod
    def category_type(cls) -> Type['Operator']: return UnaryOperator

    @abstractmethod
    def apply(self, operand: pd.DataFrame) -> pd.DataFrame: ...

    def evaluate(self) -> pd.DataFrame:
        return self.apply(self._operand.evaluate())

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand})"

    @property
    def is_feature(self): return self._operand.is_feature


class BinaryOperator(Operator):
    def __init__(self, lhs: Union[Expression, float], rhs: Union[Expression, float]) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return BinaryOperator

    @abstractmethod
    def apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame: ...

    def evaluate(self) -> pd.DataFrame:
        result = None
        try:
            result = self.apply(self._lhs.evaluate(), self._rhs.evaluate())
        except:
            print("OPPOSITE BINARY (lhs, rhs):", self._lhs, self._rhs)
            result = self.apply(self._rhs.evaluate(), self._lhs.evaluate())
        return result

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs})"

    @property
    def is_feature(self): return self._lhs.is_feature or self._rhs.is_feature

    @classmethod
    def is_rolling(cls): return False


class RollingOperator(Operator):
    def __init__(self, operand: Union[Expression, float], delta_time: Union[int, DeltaTime]) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = delta_time

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return RollingOperator

    def evaluate(self) -> pd.DataFrame:
        return self.apply(self._operand.evaluate(), self._delta_time)

    @abstractmethod
    def apply(self, data: pd.DataFrame, window: DeltaTime) -> pd.DataFrame: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand},{self._delta_time})"

    @property
    def is_feature(self): return self._operand.is_feature

    @classmethod
    def is_rolling(cls): return True

class PairRollingOperator(Operator):
    def __init__(self,
                 lhs: Expression, rhs: Expression,
                 delta_time: Union[int, DeltaTime]) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = delta_time

    @classmethod
    def n_args(cls) -> int: return 3

    @classmethod
    def category_type(cls) -> Type['Operator']: return PairRollingOperator

    @abstractmethod
    def apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame: ...

    def evaluate(self) -> pd.DataFrame:
        return self.apply(self._lhs.evaluate(), self._rhs.evaluate())

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs},{self._delta_time})"

    @property
    def is_featured(self): return self._lhs.is_feature or self._rhs.is_feature
