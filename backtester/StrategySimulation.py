import pandas as pd
from backtester.StrategyStatistics import StrategyStatistics

class StrategySimulation:
    def __init__(self, strategy_returns: pd.Series, sim_start: str, sim_end: str, is_start: str, os_start: str):
        self.sim_start = sim_start
        self.sim_end = sim_end
        self.is_start = is_start
        self.os_start = os_start
        self.strategy_is_statistics = StrategyStatistics(strategy_returns.loc[is_start:os_start])
        self.strategy_os_statistics = StrategyStatistics(strategy_returns.loc[os_start:sim_end])
        
    def get_sharpe(self, is_or_os: str):
        # Calculates the sharpe ratio for the in-sample or out-of-sample period
        if is_or_os == 'is':
            return self.strategy_is_statistics.sharpe()
        elif is_or_os == 'os':
            return self.strategy_os_statistics.sharpe()
        else:
            raise ValueError('is_or_os must be either "is" or "os"')
        
    def get_max_drawdown(self, is_or_os: str):
        # Calculates the maximum drawdown for the in-sample or out-of-sample period
        if is_or_os == 'is':
            return self.strategy_is_statistics.max_drawdown()
        elif is_or_os == 'os':
            return self.strategy_os_statistics.max_drawdown()
        else:
            raise ValueError('is_or_os must be either "is" or "os"')
        
    def get_sortino(self, is_or_os: str):
        # Calculates the sortino ratio for the in-sample or out-of-sample period
        if is_or_os == 'is':
            return self.strategy_is_statistics.sortino()
        elif is_or_os == 'os':
            return self.strategy_os_statistics.sortino()
        else:
            raise ValueError('is_or_os must be either "is" or "os"')
        
    def get_calmar(self, is_or_os: str):
        # Calculates the calmar ratio for the in-sample or out-of-sample period
        if is_or_os == 'is':
            return self.strategy_is_statistics.calmar()
        elif is_or_os == 'os':
            return self.strategy_os_statistics.calmar()
        else:
            raise ValueError('is_or_os must be either "is" or "os"')
        
    def get_cagr(self, is_or_os: str):
        # Calculates the CAGR for the in-sample or out-of-sample period
        if is_or_os == 'is':
            return self.strategy_is_statistics.cagr()
        elif is_or_os == 'os':
            return self.strategy_os_statistics.cagr()
        else:
            raise ValueError('is_or_os must be either "is" or "os"')

    def get_ic(self, is_or_os: str, strategy_weights: pd.DataFrame):
        # Calculates the information coefficient for the in-sample or out-of-sample period
        if is_or_os == 'is':
            return self.strategy_is_statistics.ic(strategy_weights)
        elif is_or_os == 'os':
            return self.strategy_os_statistics.ic(strategy_weights)
        else:
            raise ValueError('is_or_os must be either "is" or "os"')

    def get_ric(self, is_or_os: str, strategy_weights: pd.DataFrame):
        # Calculates the rank information coefficient for the in-sample or out-of-sample period
        if is_or_os == 'is':
            return self.strategy_is_statistics.ric(strategy_weights)
        elif is_or_os == 'os':
            return self.strategy_os_statistics.ric(strategy_weights)
        else:
            raise ValueError('is_or_os must be either "is" or "os"')

    def get_os_is_ratio(self):
        # Calculates the ratio of out-of-sample to in-sample sharpe ratios
        return self.get_sharpe('os') / self.get_sharpe('is')
        
