import pandas as pd
import numpy as np
from typing import Union


def compounding_returns(x: Union[pd.Series, np.array], start_value: float = 1) -> Union[pd.Series, np.array]:
    return np.cumprod(1 + x) * start_value


def center_returns(x: pd.Series, log: bool = True) -> pd.Series:
    if log:
        x = np.log(x + 1)
    return x - x.expanding().mean()