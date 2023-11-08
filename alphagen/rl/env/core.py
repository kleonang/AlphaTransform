from typing import Tuple, Optional, List
import gymnasium as gym
import math
import torch

from config import MAX_EXPR_LENGTH, SIM_START, SIM_END, TRAIN_START, TEST_START, DELAY, MIN_REWARD
from alphagen.representation.tokens import *
from alphagen.representation.tree import ExpressionBuilder
from alphagen.utils import reseed_everything
from backtester.Operator import *
from backtester.StrategyOperator import Neutralize
from backtester.StrategySimulator import StrategySimulator

class AlphaEnvCore(gym.Env):
    # _tokens: List[Token]
    # _builder: ExpressionBuilder
    # _print_expr: bool

    def __init__(self,
                 device: torch.device = torch.device('cpu'),
                 print_expr: bool = False,
                 loss_metric: str = 'Sharpe' # 'IC' or 'RIC' or 'Sharpe'
                 ):
        super().__init__()

        self._print_expr = print_expr
        self._device = device
        self._loss_metric = loss_metric

        self.eval_cnt = 0
        self.strategy_simulator = StrategySimulator(SIM_START, SIM_END, TRAIN_START, TEST_START, DELAY)
        self.render_mode = None

    def reset(
        self, *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Tuple[List[Token], dict]:
        reseed_everything(seed)
        self._tokens = [BEG_TOKEN]
        self._builder = ExpressionBuilder()
        return self._tokens, self._valid_action_types()

    def step(self, action: Token) -> Tuple[List[Token], float, bool, bool, dict]:
        if (isinstance(action, SequenceIndicatorToken) and
                action.indicator == SequenceIndicatorType.SEP):
            # Append "Neutralize" operator to end of all expressions for long-short neutrality
            neutralize = OperatorToken(Neutralize)
            self._tokens.append(neutralize)
            self._builder.add_token(neutralize)
            reward = self._evaluate()
            done = True
        elif len(self._tokens) < MAX_EXPR_LENGTH:
            try: 
                self._tokens.append(action)
                self._builder.add_token(action)
                done = False
                reward = 0.0
            except:
                done = True
                reward = MIN_REWARD[self._loss_metric]
        else:
            done = True
            if self._builder.is_valid():
                # Append "Neutralize" operator to end of all expressions for long-short neutrality
                neutralize = OperatorToken(Neutralize)
                self._tokens.append(neutralize)
                self._builder.add_token(neutralize)
                reward = self._evaluate()
            else:
                reward = MIN_REWARD[self._loss_metric]

        if math.isnan(reward):
            reward = MIN_REWARD[self._loss_metric]

        truncated = False  # For gymnasium
        return self._tokens, reward, done, truncated, self._valid_action_types()

    def _evaluate(self):
        expr: Expression = self._builder.get_tree()
        if self._print_expr:
            print(expr)
        try:
            ret = self.strategy_simulator.loss_from_expression(expr, loss=self._loss_metric)
            if math.isnan(ret):
                print(f"Casting nan to {MIN_REWARD[self._loss_metric]}")
                ret = MIN_REWARD[self._loss_metric]
            print(f"{self._loss_metric}: {ret}")
            self.eval_cnt += 1
            return ret
        except OutOfDataRangeError:
            return 0. # Change to MIN_REWARD[self._loss_metric]?

    def _valid_action_types(self) -> dict:
        valid_op_unary = self._builder.validate_op(UnaryOperator)
        valid_op_binary = self._builder.validate_op(BinaryOperator)
        valid_op_rolling = self._builder.validate_op(RollingOperator)
        valid_op_pair_rolling = self._builder.validate_op(PairRollingOperator)

        valid_op = valid_op_unary or valid_op_binary or valid_op_rolling or valid_op_pair_rolling
        valid_dt = self._builder.validate_dt()
        valid_const = self._builder.validate_const()
        valid_feature = self._builder.validate_feature()
        valid_stop = self._builder.is_valid()

        ret = {
            'select': [valid_op, valid_feature, valid_const, valid_dt, valid_stop],
            'op': {
                UnaryOperator: valid_op_unary,
                BinaryOperator: valid_op_binary,
                RollingOperator: valid_op_rolling,
                PairRollingOperator: valid_op_pair_rolling
            }
        }
        return ret

    def valid_action_types(self) -> dict:
        return self._valid_action_types()

    def render(self, mode='human'):
        pass
