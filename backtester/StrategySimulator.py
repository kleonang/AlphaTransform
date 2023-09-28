import pandas as pd
import matplotlib.pyplot as plt
from StrategyOperator import StrategyOperator
from StrategySimulation import StrategySimulation
from SimulationData import Returns, Open, Close, Low, High
from Tokenizer import Tokenizer
from StrategyTree import StrategyTree
from collections import deque

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
        returns = Returns(self.sim_start, self.sim_end, delay=self.delay).get_data()
        data = StrategyOperator.ts_mean(returns, 1)
        data = StrategyOperator.rank(data)
        data = StrategyOperator.flip(data)
        data = StrategyOperator.neutralize(data)
        return data
    
    # Build strategy tree backwards from tokens recursively and return root node
    def build_strategy_tree(self, tokens: list) -> StrategyTree:
        queue = deque(tokens)
        return self.build_strategy_tree_helper(queue)
    
    def build_strategy_tree_helper(self, queue: deque) -> StrategyTree:
        # Base case
        if len(queue) == 0:
            return None
        # Recursive case
        token = queue.pop()
        if Tokenizer.is_operator(token):
            # Create operator node
            node = StrategyTree.from_operator(token)
            # Add children to operator node
            children = []
            for _ in range(node.operator.__code__.co_argcount):
                child = self.build_strategy_tree_helper(queue)
                children.append(child)
            for child in reversed(children):
                node.add_child(child)
            return node
        elif Tokenizer.is_constant(token):
            # Create constant node
            node = StrategyTree.from_operator(token)
            return node
        elif Tokenizer.is_data(token):
            # Create data node
            node = StrategyTree.from_data(token, self.sim_start, self.sim_end, self.delay)
            return node
        else:
            raise ValueError('Invalid token: {}'.format(token))
        

    # # Build strategy tree from tokens and return root node
    # def build_strategy_tree(self, tokens: list) -> StrategyTree:
    #     # Converts a list of postorder traversal tokens into a StrategyTree object (by building a tree)
    #     # e.g. ['returns', '5', 'ts_mean', 'rank', 'flip'] -> (root: flip, children: rank), (root: rank, children: ts_mean), (root: ts_mean, children: returns, 5)
    #     # e.g. ['close', 'open', 'subtract', 'high', 'low', 'subtract', 'divide']
    #     queue = deque()
    #     # Go backwards through tokens
    #     for token in tokens:
    #         if Tokenizer.is_operator(token):
    #             # Create operator node
    #             node = StrategyTree.from_operator(token)
    #             # Add children to operator node
    #             for _ in range(node.operator.__code__.co_argcount):
    #                 child = queue.popleft()
    #                 node.add_child(child)
    #             # Add operator node to queue
    #             queue.append(node)
    #         elif Tokenizer.is_constant(token):
    #             # Create constant node
    #             node = StrategyTree.from_operator(token)
    #             # Add constant node to queue
    #             queue.append(node)
    #         elif Tokenizer.is_data(token):
    #             # Create data node
    #             node = StrategyTree.from_data(token, self.sim_start, self.sim_end, self.delay)
    #             # Add data node to queue
    #             queue.append(node)
    #         else:
    #             raise ValueError('Invalid token: {}'.format(token))
    #     # Last node in queue is the root node
    #     root = queue.pop()
    #     return root

    def generate_tree_strategy_weights(self, tokens: list) -> pd.DataFrame:
        return self.build_strategy_tree(tokens).evaluate()

    # def generate_strategy_weights(self):
    #     # Example: (Close - Open) / (High - Low)
    #     open = Open(self.sim_start, self.sim_end, delay=self.delay)
    #     close = Close(self.sim_start, self.sim_end, delay=self.delay)
    #     low = Low(self.sim_start, self.sim_end, delay=self.delay)
    #     high = High(self.sim_start, self.sim_end, delay=self.delay)
    #     open_operator = StrategyOperator(open)
    #     close_operator = StrategyOperator(close)
    #     low_operator = StrategyOperator(low)
    #     high_operator = StrategyOperator(high)
    #     close_operator.subtract(open_operator)
    #     high_operator.subtract(low_operator)
    #     close_operator.divide(high_operator)
    #     close_operator.rank()
    #     close_operator.flip()
    #     close_operator.neutralize()
    #     return close_operator.data

    def simulate_strategy_returns(self, strategy_weights: pd.DataFrame) -> pd.Series:
        # Print strategy weights
        # print("sum of positive weights:\n", strategy_weights[strategy_weights > 0].sum(axis=1).loc[self.is_start:self.sim_end], "\n")
        # print("sum of negative weights:\n", strategy_weights[strategy_weights < 0].sum(axis=1).loc[self.is_start:self.sim_end], "\n")
        print("Max weight:", round(strategy_weights.abs().max().max() * 100, 2), "%\n")

        # Calculate strategy returns
        strategy_returns = (strategy_weights * self.returns).sum(axis=1)
        strategy_returns.name = 'strategy_returns'
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
        strategy_returns = self.simulate_strategy_returns(strategy_weights)
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

    def simulate_tokens(self, tokens: list):
        strategy_weights = self.generate_tree_strategy_weights(tokens)
        strategy_returns = self.simulate_strategy_returns(strategy_weights)
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

    def loss_from_tokens(self, tokens: list, loss: str = 'IC') -> float:
        strategy_weights = self.generate_tree_strategy_weights(tokens)
        strategy_returns = self.simulate_strategy_returns(strategy_weights)
        strategy_simulation = StrategySimulation(strategy_returns, self.sim_start, self.sim_end, self.is_start, self.os_start)
        # Return loss
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
