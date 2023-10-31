import pandas as pd
import numpy as np

class TripleBarrierMethod:
    """
    Apply the Triple Barrier Method to financial returns data by Marcos López de Prado.

    Attributes:
    returns (pandas.Series): Series of returns.
    n (int): Window size for the barrier method.
    barrier (float): Barrier value to determine a label. E.g. for barrier=0.1 the label is 1 if the timeseries has a future return greater than 10% somewhere between the future 1 to n observations.
    center (bool, optional): Center the returns by their mean to denominate an above-average return. Defaults to True.
    """
    def __init__(self, returns: pd.Series, n: int, barrier: float, center: bool = True) -> None:
        self.returns = returns
        self.n = n
        self.barrier = barrier
        self.center = center

        # Use numeric values as labels (keys) to find the minimum later.
        self.conditions = {
            # Negative barriers
            -2: lambda x: (x <= -2*barrier),   
            -1: lambda x: (x > -2*barrier) & (x <= -1*barrier),
            # Neutral: No condition needed as it will be detected if no other barrier has been hit
            # Positive barriers
            1: lambda x: (x > 1*barrier) & (x <= 2*barrier),
            2: lambda x: (x > 2*barrier),
        }
        # Caching
        self._cumulative_returns: pd.DataFrame = None
        self._labels: pd.Series = None

    def clear(self):
        """
        Clear previously computed results.
        """
        self._cumulative_returns = None
        self._labels = None

    @property
    def cumulative_returns(self) -> pd.DataFrame:
        """
        Get the future cumulative returns for a range of 1 to n observations. This will be used to check which barrier has been hit first.
        Calculating if not already done.

        Returns:
        pandas.DataFrame: Cumulative returns. Shape: (self.returns.index, n).
        """
        if self._cumulative_returns is None:
            self._get_cumulative_returns()
        return self._cumulative_returns

    def _get_cumulative_returns(self) -> None:
        cumulative_returns = {}
        for i in range(1, self.n + 1):
            cumulative_returns[i] = np.exp(np.log(self.returns + 1).rolling(i).sum().shift(-i)) - 1

            if self.center:
                cumulative_returns[i] = cumulative_returns[i] - cumulative_returns[i].expanding().mean()

        self._cumulative_returns = pd.DataFrame(cumulative_returns)

    def _check_barrier_crossing(self) -> pd.DataFrame:
        """
        Check barrier crossing for future cumulative returns.

        Returns:
        pandas.DataFrame: Barrier crossing outcomes.
        """
        triple_barrier = {}
        for label, condition in self.conditions.items():
            # Check if the condition has been met inside the rolling window of cumulative returns, columns are range(1, n+1) for the cumulative return window
            triple_barrier[label] = condition(self.cumulative_returns)
            # For each barrier, get the first i in range(n) that crossed it. Replace False with np.nan to detect the first crossing by checking for not-missing values
            # columns: [-2, -1, 1, 2] as names for the respective barriers
            triple_barrier[label] = triple_barrier[label].replace(False, np.nan).apply(pd.Series.first_valid_index, axis=1)
        return pd.DataFrame(triple_barrier)

    def _remove_double_barrier_crossings(self, triple_barrier: pd.DataFrame) -> pd.DataFrame:
        """
        Remove double barrier crossings from the barrier outcomes. This happens since to cross a 2nd (positive or negative) barrier, their respective 1st barrier needs to have been crossed before.

        Args:
        triple_barrier (pandas.DataFrame): Barrier crossing outcomes.

        Returns:
        pandas.DataFrame: Updated barrier outcomes.
        """
        triple_barrier.loc[triple_barrier[2].notna(), 1] = np.nan
        triple_barrier.loc[triple_barrier[-2].notna(), -1] = np.nan
        return triple_barrier

    def _identify_barrier_hit(self, triple_barrier: pd.DataFrame) -> pd.DataFrame:
        """
        Identify barrier hits from the barrier outcomes.

        Args:
        triple_barrier (pandas.DataFrame): Barrier crossing outcomes.

        Returns:
        pandas.DataFrame: Identified barrier hits.
        """
        # .idxmin() returns np.nan if no barriers have been hit => .fillna(0) labels the "no barrier hit" condition
        triple_barrier = triple_barrier.idxmin(axis=1).fillna(0).astype(int)
        # Correct for a possible look-ahead-bias & errors introduced by .fillna()
        triple_barrier.iloc[-self.n:] = np.nan  
        return triple_barrier
    
    @property
    def labels(self) -> pd.Series:
        """
        Execute the Triple Barrier Method. Calculating if not already done.

        Returns:
        pandas.Series: Labeles for each timestep based on barrier hits.
        """
        if self._labels is None:
            self._get_labels()
        return self._labels
    
    def _get_labels(self) -> pd.Series:
        """
        Execute the Triple Barrier Method.

        Returns:
        pandas.Series: Labeles for each timestep based on barrier hits.
        """
        barriers = self._check_barrier_crossing()
        barriers = self._remove_double_barrier_crossings(barriers)
        labels = self._identify_barrier_hit(barriers)
        return labels


    def __repr__(self):
        return f"TripleBarrierMethod(returns, n={self.n}, barrier={self.barrier}, center={self.center})"
    