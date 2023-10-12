import pandas as pd
from backtester.StrategyStatistics import StrategyStatistics

class StrategySimulation:
    def __init__(self, strategy_returns: pd.Series, sim_start: str, sim_end: str, train_start: str, test_start: str):
        self.sim_start = sim_start
        self.sim_end = sim_end
        self.train_start = train_start
        self.test_start = test_start
        self.strategy_train_statistics = StrategyStatistics(strategy_returns.loc[train_start:test_start])
        self.strategy_test_statistics = StrategyStatistics(strategy_returns.loc[test_start:sim_end])
        
    def get_sharpe(self, train_or_test: str):
        # Calculates the sharpe ratio for the in-sample or out-of-sample period
        if train_or_test == 'train':
            return self.strategy_train_statistics.sharpe()
        elif train_or_test == 'test':
            return self.strategy_test_statistics.sharpe()
        else:
            raise ValueError('train_or_test must be either "is" or "os"')
        
    def get_max_drawdown(self, train_or_test: str):
        # Calculates the maximum drawdown for the in-sample or out-of-sample period
        if train_or_test == 'train':
            return self.strategy_train_statistics.max_drawdown()
        elif train_or_test == 'test':
            return self.strategy_test_statistics.max_drawdown()
        else:
            raise ValueError('train_or_test must be either "is" or "os"')
        
    def get_sortino(self, train_or_test: str):
        # Calculates the sortino ratio for the in-sample or out-of-sample period
        if train_or_test == 'train':
            return self.strategy_train_statistics.sortino()
        elif train_or_test == 'test':
            return self.strategy_test_statistics.sortino()
        else:
            raise ValueError('train_or_test must be either "is" or "os"')
        
    def get_calmar(self, train_or_test: str):
        # Calculates the calmar ratio for the in-sample or out-of-sample period
        if train_or_test == 'train':
            return self.strategy_train_statistics.calmar()
        elif train_or_test == 'test':
            return self.strategy_test_statistics.calmar()
        else:
            raise ValueError('train_or_test must be either "is" or "os"')
        
    def get_cagr(self, train_or_test: str):
        # Calculates the CAGR for the in-sample or out-of-sample period
        if train_or_test == 'train':
            return self.strategy_train_statistics.cagr()
        elif train_or_test == 'test':
            return self.strategy_test_statistics.cagr()
        else:
            raise ValueError('train_or_test must be either "is" or "os"')

    def get_ic(self, train_or_test: str, strategy_weights: pd.DataFrame):
        # Calculates the information coefficient for the in-sample or out-of-sample period
        if train_or_test == 'train':
            return self.strategy_train_statistics.ic(strategy_weights)
        elif train_or_test == 'test':
            return self.strategy_test_statistics.ic(strategy_weights)
        else:
            raise ValueError('train_or_test must be either "is" or "os"')

    def get_ric(self, train_or_test: str, strategy_weights: pd.DataFrame):
        # Calculates the rank information coefficient for the in-sample or out-of-sample period
        if train_or_test == 'train':
            return self.strategy_train_statistics.ric(strategy_weights)
        elif train_or_test == 'test':
            return self.strategy_test_statistics.ric(strategy_weights)
        else:
            raise ValueError('train_or_test must be either "is" or "os"')

    def get_os_is_ratio(self):
        # Calculates the ratio of out-of-sample to in-sample sharpe ratios
        return self.get_sharpe('test') / self.get_sharpe('train')
        
