import pandas as pd
import SimulationData
from Operator import *
import StrategyOperator
from Tokenizer import Tokenizer

# Represents a node in the strategy tree consisting of operators and data
class StrategyTree:
    # Default constructor
    def __init__(self, expression: Expression) -> None:
        self.expression = expression
        self.children: ['StrategyTree'] = []

    # Factory method for operator nodes (includes constants)
    @classmethod
    def from_operator(cls, op: str) -> 'StrategyTree':
        if Tokenizer.is_constant(op):
            operator = int(op) if op.isdigit() else float(op)
            is_constant = True
        else:
            operator = getattr(StrategyOperator, op)
            is_constant = False
        is_data = False
        return cls(operator, is_constant, is_data, op)
    
    # Factory method for data nodes
    @classmethod
    def from_data(cls, data: str, sim_start: str, sim_end: str, delay: int) -> 'StrategyTree':
        data_class = getattr(SimulationData, data.capitalize())
        operator = data_class(sim_start, sim_end, delay).get_data()
        return cls(operator, data)
    
    def evaluate(self) -> pd.DataFrame:
        # Evaluates the strategy tree and returns the resulting data
        if isinstance(self.expression, Feature):
            return self.expression._feature
        elif isinstance(self.expression, Constant):
            return self.expression._value
        # Evaluate children first
        # s = "("
        # s += self.name + ": "
        # for c in self.children:
        #     s += c.name + ", "
        # s = s[:-2]
        # s += ")"
        # print(s)
        elif isinstance(self.expression, Operator):
            evaluated_children = [child.evaluate() for child in self.children]
            # Evaluate operator
            return self.expression.apply(*evaluated_children)
        
    def add_child(self, child: 'StrategyTree') -> None:
        # Adds a child to the strategy tree
        self.children.append(child)
        