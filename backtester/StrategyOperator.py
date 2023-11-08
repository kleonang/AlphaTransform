from backtester.Operator import BinaryOperator, PairRollingOperator, UnaryOperator, RollingOperator
import pandas as pd
import numpy as np

# All operators are defined here. Class names for operators should be in PascalCase.

class Power(BinaryOperator):
    def apply(self, data: pd.DataFrame, power: pd.DataFrame) -> pd.DataFrame: 
        return data ** power
    

class Abs(UnaryOperator):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame: 
        return data.abs()
    

class Log(UnaryOperator):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame: 
        return data.apply(lambda x: np.log(x))
    

class Sigmoid(UnaryOperator):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame: 
        return 1 / (1 + np.exp(-data))
    

class Exp(UnaryOperator):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame: 
        # Convert to float128 to avoid overflow
        data = data.astype(np.float128)
        return data.apply(lambda x: np.exp(x))

class Flip(UnaryOperator):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame: 
        return -data


class Invert(UnaryOperator):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame: 
        return 1 / data
    

class Multiply(BinaryOperator): 
    def apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame: 
        return lhs * rhs
    

class Divide(BinaryOperator):
    def apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame: 
        return lhs / rhs
    

class Add(BinaryOperator):
    def apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame: 
        return lhs + rhs


class Subtract(BinaryOperator):
    def apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame: 
        return lhs - rhs
    

class TsRank(RollingOperator):
    def apply(self, data: pd.DataFrame, window: int) -> pd.DataFrame: 
        return data.rolling(window).rank(pct=True)


class TsZscore(RollingOperator):
    def apply(self, data: pd.DataFrame, window: int) -> pd.DataFrame: 
        return (data - data.rolling(window).mean()) / data.rolling(window).std()
    

class TsZscoreRank(RollingOperator):
    def apply(self, data: pd.DataFrame, window: int) -> pd.DataFrame: 
        ts_zscore = (data - data.rolling(window).mean()) / data.rolling(window).std()
        return ts_zscore.rolling(window).rank(pct=True)
    

class TsMean(RollingOperator):
    def apply(self, data: pd.DataFrame, window: int) -> pd.DataFrame: 
        return data.rolling(window).mean()
    

class TsStd(RollingOperator):
    def apply(self, data: pd.DataFrame, window: int) -> pd.DataFrame: 
        return data.rolling(window).std()
    

class TsChange(RollingOperator):
    def apply(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        return data.pct_change(periods=window, fill_method=None)
    

class TsDelta(RollingOperator):
    def apply(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        return data.diff(window)

class TsSkewness(RollingOperator):
    def apply(self, data: pd.DataFrame, window: int) -> pd.DataFrame: 
        return data.rolling(window).skew()
    

class TsKurtosis(RollingOperator):
    def apply(self, data: pd.DataFrame, window: int) -> pd.DataFrame: 
        return data.rolling(window).kurt()
    

class Rank(UnaryOperator):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame: 
        return data.rank(axis=1, pct=True)
    

class Softmax(UnaryOperator):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame: 
        return np.exp(data) / np.exp(data).sum(axis=1).values.reshape(-1, 1)
    

# class Truncate(BinaryOperator):
#     def apply(self, data: pd.DataFrame, max_percent: float) -> pd.DataFrame: 
#         max_percent_value = data.abs().sum(axis=1).values.reshape(-1, 1) * max_percent
#         return data.where(data.abs() <= max_percent_value, max_percent_value * np.sign(data))
    

class Neutralize(UnaryOperator):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame: 
        # Demean
        data = data - data.mean(axis=1).values.reshape(-1, 1)
        # Divide by sum of absolute positive and negative weights
        return data / (data.abs().sum(axis=1).values.reshape(-1, 1) / 2)
    
    
class Normalize(UnaryOperator):
    def apply(self, data: pd.DataFrame) -> pd.DataFrame: 
        # Normalizes the strategy weights (demean and divide by std dev)
        return (data - data.mean(axis=1).values.reshape(-1, 1)) / data.std(axis=1).values.reshape(-1, 1)

class Cov(PairRollingOperator):
    def apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame, window: int) -> pd.DataFrame:
        return lhs.rolling(window).cov(rhs)

class Corr(PairRollingOperator):
    def apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame, window: int) -> pd.DataFrame:
        return lhs.rolling(window).corr(rhs)
