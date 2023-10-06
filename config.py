from backtester.SimulationData import *
from backtester.StrategyOperator import *

MAX_EXPR_LENGTH = 20

# OPERATORS = [
#     # Unary
#     Abs,  # Sign,
#     Log,
#     # Binary
#     Add, Sub, Mul, Div, Greater, Less,
#     # Rolling
#     Ref, Mean, Sum, Std, Var,  # Skew, Kurt,
#     Max, Min,
#     Med, Mad,  # Rank,
#     Delta, WMA, EMA,
#     # Pair rolling
#     Cov, Corr
# ]

# OPERATORS = [op for op in dir(StrategyOperator) if 'Operator' not in op and op[0].isupper()]

OPERATORS = [ # Unary
            Abs, Log, Sigmoid, Exp, Flip, Invert, Rank, Softmax, Neutralize, Normalize,
            # Binary
            Power, Add, Subtract, Multiply, Divide, #Truncate,
            # Rolling
            TsRank, TsZscore, TsZscoreRank, TsMean, TsStd, TsChange, TsDelta, TsSkewness, TsKurtosis]

DELTA_TIMES = [10, 20, 30, 40, 50]

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

REWARD_PER_STEP = 0.

SIM_START = '2011-01-01'
SIM_END = '2017-12-07'
IS_START = '2013-04-10'
OS_START = '2017-01-03'
DELAY = 1

FEATURES = [Open(SIM_START, SIM_END, DELAY), 
            High(SIM_START, SIM_END, DELAY), 
            Low(SIM_START, SIM_END, DELAY), 
            Close(SIM_START, SIM_END, DELAY), 
            Volume(SIM_START, SIM_END, DELAY), 
            Returns(SIM_START, SIM_END, DELAY)]
