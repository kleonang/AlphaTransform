from enum import IntEnum
from typing import Type
from backtester.SimulationData import SimulationData

class SequenceIndicatorType(IntEnum):
    BEG = 0
    SEP = 1


class Token:
    def __repr__(self):
        return str(self)


class ConstantToken(Token):
    def __init__(self, constant: float) -> None:
        self.constant = constant

    def __str__(self): return str(self.constant)


class DeltaTimeToken(Token):
    def __init__(self, delta_time: int) -> None:
        self.delta_time = delta_time

    def __str__(self): return str(self.delta_time)


class FeatureToken(Token):
    def __init__(self, data: SimulationData) -> None:
        self.data = data

    def __str__(self): return '$' + self.data.__name__.lower()


class OperatorToken(Token):
    def __init__(self, operator) -> None:
        self.operator = operator

    def __str__(self): return self.operator.__name__


class SequenceIndicatorToken(Token):
    def __init__(self, indicator: SequenceIndicatorType) -> None:
        self.indicator = indicator

    def __str__(self): return self.indicator.name


BEG_TOKEN = SequenceIndicatorToken(SequenceIndicatorType.BEG)
SEP_TOKEN = SequenceIndicatorToken(SequenceIndicatorType.SEP)
