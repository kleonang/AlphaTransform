import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtester.SimulationData import Returns
from backtester.StrategyOperator import Neutralize
from alphagen.representation.tokens import FeatureToken, OperatorToken
from alphagen.representation.tree import ExpressionBuilder

class StrategyStatistics:
    # Class that provides methods for calculating quant backtesting statistics
    def __init__(self, returns_series: pd.Series):
        self.returns_series = returns_series
        tokens = [FeatureToken(Returns(returns_series.index[0], returns_series.index[-1], delay=0)), OperatorToken(Neutralize)]
        builder = ExpressionBuilder()
        for token in tokens:
            builder.add_token(token)
        self.neutralized_returns = builder.evaluate()

    def sharpe(self):
        # Calculates the sharpe ratio for strategy
        return self.returns_series.mean() / self.returns_series.std() * np.sqrt(252)
    
    def max_drawdown(self):
        # Calculates the maximum drawdown for strategy
        max_dd = (self.returns_series.cumsum() - self.returns_series.cumsum().cummax()).min()
        return max(-(max_dd), 0)
    
    def sortino(self):
        # Calculates the sortino ratio for strategy
        return self.returns_series.mean() / self.returns_series[self.returns_series < 0].std() * np.sqrt(252)
    
    def calmar(self):
        # Calculates the calmar ratio for strategy
        return self.cagr() / self.max_drawdown()
    
    def cagr(self):
        # Calculates the CAGR for strategy
        return (self.returns_series + 1).prod() ** (252 / len(self.returns_series)) - 1
    
    def ic(self, strategy_weights: pd.DataFrame):
        # Calculates the information coefficient for strategy

        # Correlation with neutralized returns
        sliced_strategy_weights = strategy_weights.loc[self.returns_series.index]
        return sliced_strategy_weights.corrwith(self.neutralized_returns).mean()
        
        # Correlation with pure returns
        # sliced_strategy_weights = strategy_weights.loc[self.returns_series.index]
        # return sliced_strategy_weights.corrwith(Returns(self.returns_series.index[0], self.returns_series.index[-1], delay=0).get_data()).mean()
    
    def ric(self, strategy_weights: pd.DataFrame):
        # Calculates the rank information coefficient for strategy

        # Correlation with neutralized returns
        sliced_strategy_weights = strategy_weights.loc[self.returns_series.index]
        return sliced_strategy_weights.corrwith(self.neutralized_returns, method='spearman').mean()
        
        # Correlation with pure returns
        # sliced_strategy_weights = strategy_weights.loc[self.returns_series.index]
        # return sliced_strategy_weights.corrwith(Returns(self.returns_series.index[0], self.returns_series.index[-1], delay=0).get_data(), method='spearman').mean()

    def plot_returns(self):
        # Plots the returns of strategy
        self.returns_series.cumsum().plot()
        plt.show()
    