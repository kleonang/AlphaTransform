import pandas as pd
import numpy as np

class StrategyOperator:
    # Class that takes in raw data and operates on it to produce strategy weights
    # def __init__(self, data: SimulationData):
    #     assert isinstance(data, SimulationData), "data must be a SimulationData object"
    #     # initialize data (e.g. close prices) for each asset for each date
    #     # data is then modified in-place by the methods below
    #     self.data = data.get_data() # rows are dates, columns are assets

    # Regular Operations
    @ staticmethod
    def power(data: pd.DataFrame, power: float):
        # Calculates the power of each asset
        return data ** power

    @ staticmethod
    def abs(data: pd.DataFrame):
        # Calculates the absolute value of each asset
        return data.abs()

    @ staticmethod
    def log(data: pd.DataFrame):
        # Calculates the log of each asset
        return data.apply(lambda x: np.log(x))

    @ staticmethod
    def sigmoid(data: pd.DataFrame):
        # Calculates the sigmoid of each asset
        return 1 / (1 + np.exp(-data))

    @ staticmethod
    def exp(data: pd.DataFrame):
        # Calculates the exponential of each asset
        return data.apply(lambda x: np.exp(x))

    @ staticmethod
    def flip(data: pd.DataFrame):
        # Calculates the negative value of each asset
        return -data

    @ staticmethod
    def invert(data: pd.DataFrame):
        # Calculates the inverse of each asset
        return 1 / data if 0 not in data.values else np.nan

    @ staticmethod
    def multiply_constant(data: pd.DataFrame, constant: float):
        # Calculates the element-wise multiplication of each asset by a constant
        return data * constant

    @ staticmethod
    def add_constant(data: pd.DataFrame, constant: float):
        # Calculates the element-wise addition of each asset by a constant
        return data + constant

    @ staticmethod
    def multiply(data: pd.DataFrame, other: pd.DataFrame):
        # Calculates the element-wise multiplication of each asset by another DataFrame
        return data * other

    @ staticmethod
    def divide(data: pd.DataFrame, other: pd.DataFrame):
        # Calculates the element-wise division of each asset by another DataFrame
        return data / other

    @ staticmethod
    def add(data: pd.DataFrame, other: pd.DataFrame):
        # Calculates the element-wise addition of each asset by another DataFrame
        return data + other

    @ staticmethod
    def subtract(data: pd.DataFrame, other: pd.DataFrame):
        # Calculates the element-wise subtraction of each asset by another DataFrame
        return data - other

    # Time Series Operations
    @ staticmethod
    def ts_rank(data: pd.DataFrame, window: int):
        # Calculates the time series rank of each asset
        return data.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

    @ staticmethod
    def ts_zscore(data: pd.DataFrame, window: int):
        # Calculates the time series z-score of each asset
        return (data - data.rolling(window).mean()) / data.rolling(window).std()

    @ staticmethod
    def ts_zscore_rank(data: pd.DataFrame, window: int):
        # Calculates the time series z-score rank of each asset
        return ts_zscore(data, window).rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

    @ staticmethod
    def ts_mean(data: pd.DataFrame, window: int):
        # Calculates the time series mean of each asset
        return data.rolling(window).mean()

    @ staticmethod
    def ts_std(data: pd.DataFrame, window: int):
        # Calculates the time series standard deviation of each asset
        return data.rolling(window).std()

    # @ staticmethod
    # def ts_change(data: pd.DataFrame, window: int, pct: bool = False):
    #     # Calculates the time series percent change of each asset
    #     return data.pct_change(window) if pct else data.diff(window)

    @ staticmethod
    def ts_skewness(data: pd.DataFrame, window: int):
        # Calculates the time series skewness of each asset
        return data.rolling(window).skew()

    @ staticmethod
    def ts_kurtosis(data: pd.DataFrame, window: int):
        # Calculates the time series kurtosis of each asset
        return data.rolling(window).kurt()

    # Cross Sectional Operations
    @ staticmethod
    def rank(data: pd.DataFrame):
        # Calculates the cross sectional rank of each asset across columns
        return data.rank(axis=1, pct=True)

    @ staticmethod
    def softmax(data: pd.DataFrame):
        # Calculates the cross sectional softmax of each asset across columns
        return np.exp(data) / np.exp(data).sum(axis=1).values.reshape(-1, 1)

    @ staticmethod
    def truncate(data: pd.DataFrame, max_percent: float = 0.01):
        # Truncates the strategy weights to a maximum percentage of sum of absolute positive and negative weights
        max_percent_value = data.abs().sum(axis=1).values.reshape(-1, 1) * max_percent
        return data.where(data.abs() <= max_percent_value, max_percent_value * np.sign(data))

    @ staticmethod
    def neutralize(data: pd.DataFrame):
        # Neutralizes the strategy weights
        # Demean
        data = data - data.mean(axis=1).values.reshape(-1, 1)
        # Divide by sum of absolute positive and negative weights
        return data / (data.abs().sum(axis=1).values.reshape(-1, 1) / 2)

    @ staticmethod
    def normalize(data: pd.DataFrame):
        # Normalizes the strategy weights (demean and divide by std dev)
        return (data - data.mean(axis=1).values.reshape(-1, 1)) / data.std(axis=1).values.reshape(-1, 1)
