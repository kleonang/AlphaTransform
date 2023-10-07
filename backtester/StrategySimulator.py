import pandas as pd
import matplotlib.pyplot as plt
from backtester.Operator import Expression
from backtester.StrategyOperator import *
from backtester.StrategySimulation import StrategySimulation
from backtester.SimulationData import Returns, Open, Close, Low, High
from alphagen.data.tokens import FeatureToken, OperatorToken, DeltaTimeToken
from alphagen.data.tree import ExpressionBuilder

class StrategySimulator:
    def __init__(self, sim_start: str, sim_end: str, is_start: str, os_start: str, delay: int = 1):
        self.sim_start = sim_start
        self.sim_end = sim_end
        self.is_start = is_start
        self.os_start = os_start
        self.delay = delay
        # Load pre-calculated true returns
        self.returns = Returns(self.sim_start, self.sim_end, delay=0).get_data()
    
    # Function to be replicated by Symbolic Regression Transformer
    def generate_strategy_weights(self) -> pd.DataFrame:
        # Example of a strategy: 3-day mean reversion
        tokens = [
            FeatureToken(Returns(self.sim_start, self.sim_end, delay=self.delay)),
            DeltaTimeToken(3),
            OperatorToken(TsMean),
            OperatorToken(Rank),
            OperatorToken(Flip),
            OperatorToken(Neutralize)
        ]
        builder = ExpressionBuilder()
        for token in tokens:
            builder.add_token(token)
        return builder.evaluate()
        
    def generate_tree_strategy_weights(self, tokens: list) -> pd.DataFrame:
        return self.build_strategy_tree(tokens).evaluate()

    def simulate_strategy_returns(self, strategy_weights: pd.DataFrame, verbose: bool = False) -> pd.Series:
        # Calculate strategy returns
        strategy_returns = (strategy_weights * self.returns).sum(axis=1)
        strategy_returns.name = 'strategy_returns'

        if verbose:
            # Print strategy weights
            # print("sum of positive weights:\n", strategy_weights[strategy_weights > 0].sum(axis=1).loc[self.is_start:self.sim_end], "\n")
            # print("sum of negative weights:\n", strategy_weights[strategy_weights < 0].sum(axis=1).loc[self.is_start:self.sim_end], "\n")
            print("Max weight:", round(strategy_weights.abs().max().max() * 100, 2), "%\n")
            print("strat returns:", strategy_returns.loc[self.is_start:self.sim_end], "\n")

            # Plot strategy returns with is in green and os in red
            plt.figure(figsize=(16,9))
            cumulative_returns = strategy_returns.loc[self.is_start:self.sim_end].cumsum()
            cumulative_returns.loc[self.is_start:self.os_start].plot(color='green')
            cumulative_returns.loc[self.os_start:self.sim_end].plot(color='red')
            plt.ylabel('returns')
            plt.title('Strategy Returns')
            plt.show()
        
        return strategy_returns
    
    def simulate(self):
        strategy_weights = self.generate_strategy_weights()
        strategy_returns = self.simulate_strategy_returns(strategy_weights, verbose=True)
        strategy_simulation = StrategySimulation(strategy_returns, self.sim_start, self.sim_end, self.is_start, self.os_start)
        # Print simulation statistics
        print('In-Sample Sharpe Ratio: {}'.format(strategy_simulation.get_sharpe('is')))
        print('Out-of-Sample Sharpe Ratio: {}'.format(strategy_simulation.get_sharpe('os')))
        print('OS/IS Ratio: {}'.format(strategy_simulation.get_os_is_ratio()))
        print()
        print('In-Sample IC: {}'.format(strategy_simulation.get_ic('is', strategy_weights)))
        print('Out-of-Sample IC: {}'.format(strategy_simulation.get_ic('os', strategy_weights)))
        print('In-Sample RIC: {}'.format(strategy_simulation.get_ric('is', strategy_weights)))
        print('Out-of-Sample RIC: {}'.format(strategy_simulation.get_ric('os', strategy_weights)))
        print('In-Sample CAGR: {}'.format(strategy_simulation.get_cagr('is')))
        print('Out-of-Sample CAGR: {}'.format(strategy_simulation.get_cagr('os')))
        print('In-Sample Max Drawdown: {}'.format(strategy_simulation.get_max_drawdown('is')))
        print('Out-of-Sample Max Drawdown: {}'.format(strategy_simulation.get_max_drawdown('os')))
        print('In-Sample Sortino Ratio: {}'.format(strategy_simulation.get_sortino('is')))
        print('Out-of-Sample Sortino Ratio: {}'.format(strategy_simulation.get_sortino('os')))
        print('In-Sample Calmar Ratio: {}'.format(strategy_simulation.get_calmar('is')))
        print('Out-of-Sample Calmar Ratio: {}'.format(strategy_simulation.get_calmar('os')))
        print()

    def loss_from_expression(self, tree: Expression, loss: str = 'IC') -> float:
        strategy_weights = tree.evaluate()
        strategy_returns = self.simulate_strategy_returns(strategy_weights)
        strategy_simulation = StrategySimulation(strategy_returns, self.sim_start, self.sim_end, self.is_start, self.os_start)
        # Return loss
        if not isinstance(strategy_weights, pd.DataFrame):
            return -1
        if loss == "IC":
            return strategy_simulation.get_ic('is', strategy_weights)
        elif loss == "RIC":
            return strategy_simulation.get_ric('is', strategy_weights)
        else:
            raise ValueError("Invalid loss function: {}".format(loss))

if __name__ == '__main__':
    # Set simulation start and end dates
    sim_start = '2011-01-01'
    sim_end = '2017-12-07'
    # Set in-sample and out-of-sample start dates
    is_start = '2013-04-10'
    os_start = '2017-01-03'
    # Initialize simulator
    strategy_simulator = StrategySimulator(sim_start, sim_end, is_start, os_start)
    # Simulate strategy
    strategy_simulator.simulate()
