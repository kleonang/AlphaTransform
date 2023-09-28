from abc import ABC
import pandas as pd
import pickle

class SimulationData(ABC):
    """
    Base class for strategy simulation data.
    """
    def __init__(self, sim_start: str, sim_end: str, delay: int = 1, data_path: str = None):
        self.data = pickle.load(open(data_path, 'rb'))
        self.data = self.data.loc[sim_start:sim_end]
        self.data = self.data.shift(delay)
    
    def get_data(self) -> pd.DataFrame:
        return self.data

# OHLCV data
class Open(SimulationData):
    """
    Class for open prices.
    """
    def __init__(self, sim_start: str, sim_end: str, delay: int = 1, data_path: str = "../data/open.pickle"):
        super().__init__(sim_start, sim_end, delay, data_path)

class High(SimulationData):
    """
    Class for high prices.
    """
    def __init__(self, sim_start: str, sim_end: str, delay: int = 1, data_path: str = "../data/high.pickle"):
        super().__init__(sim_start, sim_end, delay, data_path)

class Low(SimulationData):
    """
    Class for low prices.
    """
    def __init__(self, sim_start: str, sim_end: str, delay: int = 1, data_path: str = "../data/low.pickle"):
        super().__init__(sim_start, sim_end, delay, data_path)

class Close(SimulationData):
    """
    Class for close prices.
    """
    def __init__(self, sim_start: str, sim_end: str, delay: int = 1, data_path: str = "../data/close.pickle"):
        super().__init__(sim_start, sim_end, delay, data_path)

class Volume(SimulationData):
    """
    Class for volume.
    """
    def __init__(self, sim_start: str, sim_end: str, delay: int = 1, data_path: str = "../data/volume.pickle"):
        super().__init__(sim_start, sim_end, delay, data_path)

class Returns(SimulationData):
    """
    Class for returns.
    """
    def __init__(self, sim_start: str, sim_end: str, delay: int = 1, data_path: str = "../data/returns.pickle"):
        super().__init__(sim_start, sim_end, delay, data_path)
