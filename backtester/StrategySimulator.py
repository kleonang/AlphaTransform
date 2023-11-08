import pandas as pd
import matplotlib.pyplot as plt
from config import MIN_REWARD
from backtester.Operator import Expression
from backtester.StrategyOperator import *
from backtester.StrategySimulation import StrategySimulation
from backtester.SimulationData import Returns, Open, Close, Low, High
from alphagen.representation.tokens import FeatureToken, OperatorToken, DeltaTimeToken, Token
from alphagen.representation.tree import ExpressionBuilder

class StrategySimulator:
    def __init__(self, sim_start: str, sim_end: str, train_start: str, test_start: str, delay: int = 1):
        self.sim_start = sim_start
        self.sim_end = sim_end
        self.train_start = train_start
        self.test_start = test_start
        self.delay = delay
        # Load pre-calculated true returns
        self.returns = Returns(self.sim_start, self.sim_end, delay=0).get_data()
    
    # Function to be replicated by alpha generator
    def generate_strategy_weights(self) -> pd.DataFrame:
        # Example of a strategy: 3-day mean reversion
        tokens = [
            FeatureToken(Returns(self.sim_start, self.sim_end, delay=self.delay)),
            DeltaTimeToken(1),
            OperatorToken(TsMean),
            OperatorToken(Rank),
            OperatorToken(Flip),
            OperatorToken(Neutralize)
        ]
        builder = ExpressionBuilder()
        for token in tokens:
            builder.add_token(token)
        return builder.evaluate()

    def generate_strategy_weights_from_tokens(self, tokens: 'list[Token]') -> pd.DataFrame:
        builder = ExpressionBuilder()
        for token in tokens:
            builder.add_token(token)
        return builder.evaluate()

    def simulate_strategy_returns(self, strategy_weights: pd.DataFrame, verbose: bool = False) -> pd.Series:
        # Calculate strategy returns
        strategy_returns = (strategy_weights * self.returns).sum(axis=1)
        strategy_returns.name = 'strategy_returns'

        if verbose:
            # Print strategy weights
            # print("sum of positive weights:\n", strategy_weights[strategy_weights > 0].sum(axis=1).loc[self.is_start:self.sim_end], "\n")
            # print("sum of negative weights:\n", strategy_weights[strategy_weights < 0].sum(axis=1).loc[self.is_start:self.sim_end], "\n")
            print("Max weight:", round(strategy_weights.abs().max().max() * 100, 2), "%\n")
            print("strat returns:", strategy_returns.loc[self.train_start:self.sim_end], "\n")

            # Plot strategy returns with train period in green and test period in red
            plt.figure(figsize=(16,9))
            cumulative_returns = (strategy_returns.loc[self.train_start:self.sim_end] + 1).cumprod() - 1
            cumulative_returns.loc[self.train_start:self.test_start].plot(color='green')
            cumulative_returns.loc[self.test_start:self.sim_end].plot(color='red')

            # Use log scale for y axis if cumulative returns are greater than 100,000%
            if cumulative_returns.max() > 1000:
                plt.yscale('log')

            plt.ylabel('returns')
            plt.title('Strategy Returns')
            plt.show()
        
        return strategy_returns
    
    def simulate(self):
        strategy_weights = self.generate_strategy_weights()
        strategy_returns = self.simulate_strategy_returns(strategy_weights, verbose=True)
        strategy_simulation = StrategySimulation(strategy_returns, self.sim_start, self.sim_end, self.train_start, self.test_start)
        # Print simulation statistics
        print('Train Sharpe Ratio: {}'.format(strategy_simulation.get_sharpe('train')))
        print('Test Sharpe Ratio: {}'.format(strategy_simulation.get_sharpe('test')))
        print('Test/Train Ratio: {}'.format(strategy_simulation.get_test_train_ratio()))
        print()
        print('Train IC: {}'.format(strategy_simulation.get_ic('train', strategy_weights)))
        print('Test IC: {}'.format(strategy_simulation.get_ic('test', strategy_weights)))
        print('Train RIC: {}'.format(strategy_simulation.get_ric('train', strategy_weights)))
        print('Test RIC: {}'.format(strategy_simulation.get_ric('test', strategy_weights)))
        print('Train CAGR: {}'.format(strategy_simulation.get_cagr('train')))
        print('Test CAGR: {}'.format(strategy_simulation.get_cagr('test')))
        print('Train Max Drawdown: {}'.format(strategy_simulation.get_max_drawdown('train')))
        print('Test Max Drawdown: {}'.format(strategy_simulation.get_max_drawdown('test')))
        print('Train Sortino Ratio: {}'.format(strategy_simulation.get_sortino('train')))
        print('Test Sortino Ratio: {}'.format(strategy_simulation.get_sortino('test')))
        print('Train Calmar Ratio: {}'.format(strategy_simulation.get_calmar('train')))
        print('Test Calmar Ratio: {}'.format(strategy_simulation.get_calmar('test')))
        print()

    def simulate_tokens(self, tokens: 'list[Token]'):
        strategy_weights = self.generate_strategy_weights_from_tokens(tokens)
        strategy_returns = self.simulate_strategy_returns(strategy_weights, verbose=True)
        strategy_simulation = StrategySimulation(strategy_returns, self.sim_start, self.sim_end, self.train_start, self.test_start)
        # Print simulation statistics
        print('Train Sharpe Ratio: {}'.format(strategy_simulation.get_sharpe('train')))
        print('Test Sharpe Ratio: {}'.format(strategy_simulation.get_sharpe('test')))
        print('Test/Train Ratio: {}'.format(strategy_simulation.get_test_train_ratio()))
        print()
        print('Train IC: {}'.format(strategy_simulation.get_ic('train', strategy_weights)))
        print('Test IC: {}'.format(strategy_simulation.get_ic('test', strategy_weights)))
        print('Train RIC: {}'.format(strategy_simulation.get_ric('train', strategy_weights)))
        print('Test RIC: {}'.format(strategy_simulation.get_ric('test', strategy_weights)))
        print('Train CAGR: {}'.format(strategy_simulation.get_cagr('train')))
        print('Test CAGR: {}'.format(strategy_simulation.get_cagr('test')))
        print('Train Max Drawdown: {}'.format(strategy_simulation.get_max_drawdown('train')))
        print('Test Max Drawdown: {}'.format(strategy_simulation.get_max_drawdown('test')))
        print('Train Sortino Ratio: {}'.format(strategy_simulation.get_sortino('train')))
        print('Test Sortino Ratio: {}'.format(strategy_simulation.get_sortino('test')))
        print('Train Calmar Ratio: {}'.format(strategy_simulation.get_calmar('train')))
        print('Test Calmar Ratio: {}'.format(strategy_simulation.get_calmar('test')))
        print()

    def loss_from_expression(self, tree: Expression, loss: str = 'IC', train_or_test: str = 'train') -> float:
        strategy_weights = tree.evaluate()
        strategy_returns = self.simulate_strategy_returns(strategy_weights)
        strategy_simulation = StrategySimulation(strategy_returns, self.sim_start, self.sim_end, self.train_start, self.test_start)
        # Return loss
        try:
            if loss == "IC":
                return strategy_simulation.get_ic(train_or_test, strategy_weights)
            elif loss == "RIC":
                return strategy_simulation.get_ric(train_or_test, strategy_weights)
            elif loss == "Sharpe":
                return strategy_simulation.get_sharpe(train_or_test)
            else:
                raise ValueError("Invalid loss function: {}".format(loss))
        except:
            return MIN_REWARD[loss]

if __name__ == '__main__':
    # Set simulation start and end dates
    sim_start = '2011-01-01'
    sim_end = '2017-12-07'
    # Set Train and Test start dates
    train_start = '2013-04-10'
    test_start = '2017-01-03'
    # Initialize simulator
    strategy_simulator = StrategySimulator(sim_start, sim_end, train_start, test_start)
    # Simulate strategy
    strategy_simulator.simulate()
