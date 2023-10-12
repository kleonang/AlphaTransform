from backtester.Operator import *
from alphagen.representation.tokens import *
from backtester.SimulationData import *
from backtester.StrategyOperator import *
from typing import List

class ExpressionBuilder:
    def __init__(self):
        self.stack: List[Expression] = []

    def get_tree(self) -> Expression:
        if len(self.stack) == 1:
            return self.stack[0]
        else:
            raise InvalidExpressionException(f"Expected only one tree, got {len(self.stack)}")

    def add_token(self, token: Token):
        if not self.validate(token):
            raise InvalidExpressionException(f"Token {token} not allowed here, stack: {self.stack}.")
        if isinstance(token, OperatorToken):
            n_args: int = token.operator.n_args()
            children = []
            for _ in range(n_args):
                children.append(self.stack.pop())
            self.stack.append(token.operator(*reversed(children)))  # type: ignore
        elif isinstance(token, ConstantToken):
            self.stack.append(Constant(token.constant))
        elif isinstance(token, DeltaTimeToken):
            self.stack.append(DeltaTime(token.delta_time))
        elif isinstance(token, FeatureToken):
            self.stack.append(Feature(token.data))
        else:
            assert False

    def is_valid(self) -> bool:
        return len(self.stack) == 1 and self.stack[0].is_feature

    def validate(self, token: Token) -> bool:
        if isinstance(token, OperatorToken):
            return self.validate_op(token.operator)
        elif isinstance(token, DeltaTimeToken):
            return self.validate_dt()
        elif isinstance(token, ConstantToken):
            return self.validate_const()
        elif isinstance(token, FeatureToken):
            return self.validate_feature()
        else:
            assert False

    def validate_op(self, op: Type[Operator]) -> bool:
        if len(self.stack) < op.n_args():
            return False

        if issubclass(op, UnaryOperator):
            if not self.stack[-1].is_feature:
                return False
        elif issubclass(op, BinaryOperator):
            if not self.stack[-1].is_feature and not self.stack[-2].is_feature:
                return False
            if (isinstance(self.stack[-1], DeltaTime) or
                    isinstance(self.stack[-2], DeltaTime)):
                return False
        elif issubclass(op, RollingOperator):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_feature:
                return False
        elif issubclass(op, PairRollingOperator):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_feature or not self.stack[-3].is_feature:
                return False
        else:
            assert False
        return True

    def validate_dt(self) -> bool:
        return len(self.stack) > 0 and self.stack[-1].is_feature

    def validate_const(self) -> bool:
        return len(self.stack) == 0 or self.stack[-1].is_feature

    def validate_feature(self) -> bool:
        return not (len(self.stack) >= 1 and isinstance(self.stack[-1], DeltaTime))
    
    def evaluate(self) -> pd.DataFrame:
        return self.get_tree().evaluate()


class InvalidExpressionException(ValueError):
    pass


if __name__ == '__main__':
    # Set simulation start and end dates
    sim_start = '2011-01-01'
    sim_end = '2017-12-07'

    tokens = [
        FeatureToken(Low(sim_start, sim_end)),
        OperatorToken(Abs),
        DeltaTimeToken(10),
        OperatorToken(TsDelta),
        FeatureToken(High(sim_start, sim_end)),
        FeatureToken(Close(sim_start, sim_end)),
        OperatorToken(Divide),
        OperatorToken(Add),
    ]

    builder = ExpressionBuilder()
    for token in tokens:
        builder.add_token(token)

    print(f'res: {str(builder.get_tree())}')
    print(f'ref: Add(TsDelta(Abs($low),10),Divide($high,$close))')
