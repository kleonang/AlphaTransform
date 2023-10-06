import StrategyOperator
import SimulationData

# Tokenizes a string into a list of tokens (functions, variables, operators, etc.)
class Tokenizer():
    def tokenize(self, string: str) -> list:
        # Tokenizes a string into a list of tokens in Reverse Polish Notation (RPN)
        # e.g. 'flip(rank(ts_mean(returns, 5)))' -> ['returns', '5', 'ts_mean', 'rank', 'flip']
        pass

    @staticmethod
    def is_operator(token: str) -> bool:
        # Returns True if token is an operator from StrategyOperator.py
        operators = [fn for fn in dir(StrategyOperator) if callable(getattr(StrategyOperator, fn)) and not fn.startswith('__')]
        return token in operators
    
    @staticmethod
    def is_data(token: str) -> bool:
        # Returns True if token is a data class from SimulationData.py
        data_classes = [cls for cls in dir(SimulationData) if callable(getattr(SimulationData, cls)) and not cls.startswith('__')]
        # Filter out non-subclasses of SimulationData
        data_classes = [cls.lower() for cls in data_classes if cls != 'SimulationData' and issubclass(getattr(SimulationData, cls), SimulationData.SimulationData)]
        return token in data_classes
    
    @staticmethod
    def is_constant(token: str) -> bool:
        # Returns True if token is a constant
        return token.replace('.', '', 1).isdigit()
