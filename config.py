from backtester.SimulationData import *
from backtester.StrategyOperator import *

MAX_EXPR_LENGTH = 20

# OPERATORS = [op for op in dir(StrategyOperator) if 'Operator' not in op and op[0].isupper()]

OPERATORS = [ # Unary
            Abs, Log, Sigmoid, Exp, Flip, Invert, Rank, Softmax, Neutralize, Normalize,
            # Binary
            Power, Add, Subtract, Multiply, Divide, 
            # Rolling
            TsRank, TsZscore, TsZscoreRank, TsMean, TsStd, TsChange, TsDelta, TsSkewness, TsKurtosis,
            # PairRolling
            Cov, Corr]

DELTA_TIMES = [1, 3, 5, 10, 20, 60, 90, 120, 180, 252]

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

REWARD_PER_STEP = 0.
MIN_REWARD = {'IC': -1.,
              'RIC': -1.,
              'Sharpe': -10.}

SIM_START = '2010-01-01'
SIM_END = '2017-12-07'
TRAIN_START = '2011-01-01'
TEST_START = '2017-01-01'
DELAY = 1

FEATURES = [Open, High, Low, Close, Volume, Returns]
FEATURES = [feature(SIM_START, SIM_END, DELAY) for feature in FEATURES]
