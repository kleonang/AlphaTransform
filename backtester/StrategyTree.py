import pandas as pd
import SimulationData
from StrategyOperator import StrategyOperator
from Tokenizer import Tokenizer

# Represents a node in the strategy tree consisting of operators and data
class StrategyTree():
    # Default constructor
    def __init__(self, operator, is_constant: bool, is_data: bool, name: str) -> None:
        self.operator = operator
        self.is_constant = is_constant
        self.is_data = is_data
        self.name = name
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
        is_constant = False
        is_data = True
        return cls(operator, is_constant, is_data, data)
    
    def evaluate(self) -> pd.DataFrame:
        # Evaluates the strategy tree and returns the resulting data
        if self.is_data or self.is_constant:
            return self.operator
        # Evaluate children first
        # s = "("
        # s += self.name + ": "
        # for c in self.children:
        #     s += c.name + ", "
        # s = s[:-2]
        # s += ")"
        # print(s)
        evaluated_children = [child.evaluate() for child in self.children]
        # Evaluate operator
        return self.operator(*evaluated_children)
        
    def add_child(self, child: 'StrategyTree') -> None:
        # Adds a child to the strategy tree
        self.children.append(child)
        