import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from .barrier_conditions import BarrierConditions


class BarrierMethod:
    """
    Apply the Barrier Method to financial returns data inspired by Marcos López de Prado.

    The goal is to vectorize this method with pandas/numpy to reduce for-loops and add additional functionality:
    - multiple barriers: Instead of having a barrier at +/-1%, we have a 2nd, 3rd,... at +/-2%, +/-3% etc.
    - intermediate steps are returned as a pd.DataFrame that can be used for features of models:
        - cumulative_returns
        - time_since_last_crossing
    - transition probabilities of the generated labels
    - plot_at_date as plotting capabilities and a quick sanity check

    Attributes:
    returns (pandas.Series): Series of returns.
    n (int): Window size for the barrier method.
    barrier (float): Barrier value to determine a label. E.g. for barrier=0.1 the label is 1 if the
                     timeseries has a future return greater than 10% somewhere between the future 1 to n observations.
    center (bool, optional): Center the returns by their mean to denominate an above-average return.
                             Defaults to True.
    """

    def __init__(self, returns: pd.Series, n: int, barrier: float, n_barriers: int = 2, center: bool = True):
        self.returns = returns
        self.n = int(n)
        self.barrier = barrier
        self.center = center
        self.n_barriers = int(n_barriers)
        self.conditions = BarrierConditions(n=self.n_barriers, barrier=barrier).conditions
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
        Get the future cumulative returns for a range of 1 to n observations.
        This will be used to check which barrier has been hit first.
        Calculating if not already done.

        Returns:
        pandas.DataFrame: Cumulative returns. Shape: (self.returns.index, n).
        """
        if self._cumulative_returns is None:
            self._cumulative_returns = self._get_cumulative_returns()
        return self._cumulative_returns

    def _get_cumulative_returns(self) -> pd.DataFrame:
        cumulative_returns = {}
        for i in range(1, self.n + 1):
            cumulative_returns[i] = np.exp(np.log(self.returns + 1).rolling(i).sum().shift(-i)) - 1

            if self.center:
                cumulative_returns[i] = cumulative_returns[i] - cumulative_returns[i].expanding().mean()

        return pd.DataFrame(cumulative_returns)

    def _check_barrier_crossing(self) -> pd.DataFrame:
        """
        Check barrier crossing for future cumulative returns.

        Returns:
        pandas.DataFrame: Barrier crossing outcomes.
        """
        barrier = {}
        for label, condition in self.conditions.items():
            # Check if the condition has been met inside the rolling window of cumulative returns,
            # Columns are range(1, n+1) for the cumulative return window
            barrier[label] = condition.evaluate(self.cumulative_returns)
            # For each barrier, get the first i in range(n) that crossed it.
            # Replace False with np.nan to detect the first crossing by checking for not-missing values
            # columns: [-2, -1, 1, 2] as names for the respective barriers
            barrier[label] = (barrier[label]
                              .replace(False, np.nan)
                              .apply(pd.Series.first_valid_index, axis=1)
                              )
        return pd.DataFrame(barrier)

    def _remove_double_barrier_crossings(self, barrier: pd.DataFrame) -> pd.DataFrame:
        """
        Remove double barrier crossings from the barrier outcomes. This happens since to cross
        a 2nd (positive or negative) barrier, their respective 1st barrier needs to have been
        crossed before.

        Args:
        barrier (pandas.DataFrame): Barrier crossing outcomes.

        Returns:
        pandas.DataFrame: Updated barrier outcomes.
        """
        if self.n_barriers > 1:
            # Make a copy of the DataFrame be able to check the changes after using this function
            barrier_filtered = barrier.copy()
            for i in range(self.n_barriers, 1, -1):
                barrier_filtered.loc[barrier[i].notna(), i - 1] = np.nan
                barrier_filtered.loc[barrier[-i].notna(), -i + 1] = np.nan
            return barrier_filtered
        else:
            return barrier

    def _identify_barrier_hit(self, barrier: pd.DataFrame) -> pd.Series:
        """
        Identify barrier hits from the barrier outcomes.

        Args:
        barrier (pandas.DataFrame): Barrier crossing outcomes.

        Returns:
        pandas.DataFrame: Identified barrier hits.
        """
        # pd.Series([None]).idxmax(skipna=True) now raises a FutureWarning and will throw a ValueError eventually.
        # The solution is overly verbose, and we hope for something better.
        # See: https://github.com/pandas-dev/pandas/pull/54226#issuecomment-1794973298
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # .idxmin() returns np.nan if no barriers have been hit
            # => .fillna(0) labels the "no barrier hit" condition
            barrier = barrier.idxmin(axis=1, skipna=True).fillna(0).astype(int)
        # Correct for a possible look-ahead-bias & errors introduced by .fillna()
        barrier.iloc[-self.n:] = np.nan
        return barrier

    @property
    def labels(self) -> pd.Series:
        """
        Execute the Barrier Method. Calculating if not already done.

        Returns:
        pandas.Series: Labeles for each timestep based on barrier hits.
        """
        if self._labels is None:
            self._labels = self._get_labels()
        return self._labels

    def _get_labels(self) -> pd.Series:
        """
        Execute the Barrier Method.

        Returns:
        pandas.Series: Labeles for each timestep based on barrier hits.
        """
        barriers = self._check_barrier_crossing()
        barriers = self._remove_double_barrier_crossings(barriers)
        labels = self._identify_barrier_hit(barriers)
        return labels

    def __repr__(self):
        return f"BarrierMethod(returns, n={self.n}, barrier={self.barrier}, n_barriers={self.n_barriers}, center={self.center})"

    @property
    def time_since_last_crossing(self) -> pd.DataFrame:
        """
        Computes the time elapsed since the last crossing for each barrier in the dataset.
        This might be a useful feature for models.

        Returns:
        pd.DataFrame: A DataFrame containing the time since the last crossing for each barrier.
        """
        barrier_crossings = self._check_barrier_crossing().shift(self.n)
        crossing_seen = barrier_crossings.isna()
        crossing_groups = (crossing_seen != crossing_seen.shift()).cumsum()

        time_since_last_crossing = {}
        for label in barrier_crossings.columns:
            time_since_last_crossing[label] = crossing_seen[label].groupby(crossing_groups[label]).cumsum()
        time_since_last_crossing = pd.DataFrame(time_since_last_crossing)
        return time_since_last_crossing

    @property
    def transition_probabilities(self) -> pd.DataFrame:
        """
        Computes the transition probabilities between the different labels by dividing the frequency of transitions
        by the total count in each state.

        Raises:
        AttributeError: If the 'labels' attribute has not been calculated yet. Run .labels to compute it.

        Returns:
        pd.DataFrame: A DataFrame containing transition probabilities between different states in the dataset.
        """
        if self._labels is None:
            raise AttributeError("The attribute 'labels' have not been calculated yet. Run .labels")
        transition_counts = self.labels.groupby([self.labels.rename("From"), self.labels.shift(-1).rename("To")]).size()

        transition_probas = transition_counts.groupby(level=0).apply(lambda x: x / x.sum())
        # Bring it into a proper form
        transition_probas = transition_probas.unstack().droplevel(1).fillna(0)
        transition_probas.index = transition_probas.index.astype(int)
        transition_probas.columns = transition_probas.columns.astype(int)
        return transition_probas

    @property
    def signals_pa(self, factor: int = 252) -> float:
        """
        Calculate the annual frequency of label changes in the dataset

        Parameters:
        - factor (int, optional): Number of observations per year. Default is 252 (trading days).

        Returns:
        - float: Number of signals (changes) per year, based on the frequency of label changes.
        """
        return (self.labels != self.labels.shift()).dropna().sum() / self.labels.dropna().shape[0] * 252

    def plot_at_date(self, date: str, months=3, figsize=(12, 3)) -> None:
        """
        Plots the Barrier Method for a specified date.

        Args:
        date (str): The specific date in string format ('YYYY-MM-DD') for which to generate the plot.
        months (int): Number of months to consider before and after the given date for the plot. Default is 3 months.
        figsize (tuple): Tuple defining the width and height of the plot. Default is (12, 3).

        Raises:
        AttributeError: If the 'labels' attribute has not been calculated yet. Run .labels to compute it.

        Returns:
        None
        """
        if self._labels is None:
            raise AttributeError("The attribute 'labels' have not been calculated yet. Run .labels")

        date = pd.Timestamp(date)
        start_date = date - pd.DateOffset(months=months)
        end_date = date + pd.DateOffset(months=months)

        # Create data to plot
        index = np.cumprod(1 + self.returns.loc[start_date:end_date])
        # Issue a warning if the selected date is not present in the index
        if date not in index.index:
            warnings.warn_explicit(
                f"The selected date '{date.strftime('%Y-%m-%d')}' is not inside the attribute 'returns'." +
                f"The next available date is '{index[date:].index.min().strftime('%Y-%m-%d')}'.",
                UserWarning, "", 1
            )

            date = index[date:].index.min()
        # Normalize data for plotting
        index = index / index.loc[date] * 100

        barrier_idx = index.loc[date:].head(self.n).index
        barriers = []
        for key in self.conditions.keys():
            barriers.append(pd.Series(100 * (1 + key * self.barrier), index=barrier_idx).rename(key))
        barriers = pd.concat(barriers, axis=1)

        # Plotting
        fig, ax = plt.subplots(figsize=figsize)

        aut_locator = mdates.AutoDateLocator(minticks=months * 2)
        aut_formatter = mdates.ConciseDateFormatter(aut_locator)

        plt.axhline(100, c="grey", ls="--", lw=0.5)
        plt.title(
            f"Barrier Method for {self.returns.name} at {date.strftime('%d %b, %Y')} (label = {int(self.labels.loc[date])})")
        index.plot(ax=ax)

        text_date = barriers.index.max() + pd.DateOffset(1)
        text_spacing_y = 0.5
        for col in barriers.columns:
            c = "green" if col > 0 else "red"
            barriers[col].plot(ax=ax, c=c)
            ax.text(x=text_date, y=barriers[col].min() - text_spacing_y, s=col, style='normal', c=c)
        ax.text(x=text_date, y=100 - text_spacing_y, s=0, c="black")
        plt.plot([barriers.index.min(), barriers.index.min()],
                 [barriers[-self.n_barriers].min(), barriers[self.n_barriers].max()], c="grey", ls="dotted")
        plt.plot([barriers.index.max(), barriers.index.max()],
                 [barriers[-self.n_barriers].min(), barriers[self.n_barriers].max()], c="black")
        ax.xaxis.set_major_locator(aut_locator)
        ax.xaxis.set_major_formatter(aut_formatter)
        ax.set_xlabel(None)
        plt.xlim(start_date, end_date)
        plt.show()
