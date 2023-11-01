import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

class TripleBarrierMethod:
    """
    Apply the Triple Barrier Method to financial returns data by Marcos López de Prado.

    Attributes:
    returns (pandas.Series): Series of returns.
    n (int): Window size for the barrier method.
    barrier (float): Barrier value to determine a label. E.g. for barrier=0.1 the label is 1 if the
                     timeseries has a future return greater than 10% somewhere between the future 1 to n observations.
    center (bool, optional): Center the returns by their mean to denominate an above-average return.
                             Defaults to True.
    """

    def __init__(self, returns: pd.Series, n: int, barrier: float, center: bool = True):
        self.returns = returns
        self.n = n
        self.barrier = barrier
        self.center = center

        # Use numeric values as labels (keys) to find the minimum later.
        self.conditions = {
            # Negative barriers
            -2: lambda x: (x <= -2 * barrier),
            -1: lambda x: (x > -2 * barrier) & (x <= -1 * barrier),
            # Neutral: No condition needed as it will be detected if no other barrier has been hit
            # Positive barriers
            1: lambda x: (x > 1 * barrier) & (x <= 2 * barrier),
            2: lambda x: (x > 2 * barrier),
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
        Get the future cumulative returns for a range of 1 to n observations.
        This will be used to check which barrier has been hit first.
        Calculating if not already done.

        Returns:
        pandas.DataFrame: Cumulative returns. Shape: (self.returns.index, n).
        """
        if self._cumulative_returns is None:
            self._cumulative_returns = self._get_cumulative_returns()
        return self._cumulative_returns

    def _get_cumulative_returns(self) -> None:
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
        triple_barrier = {}
        for label, condition in self.conditions.items():
            # Check if the condition has been met inside the rolling window of cumulative returns,
            # Columns are range(1, n+1) for the cumulative return window
            triple_barrier[label] = condition(self.cumulative_returns)
            # For each barrier, get the first i in range(n) that crossed it.
            # Replace False with np.nan to detect the first crossing by checking for not-missing values
            # columns: [-2, -1, 1, 2] as names for the respective barriers
            triple_barrier[label] = (triple_barrier[label]
                                     .replace(False, np.nan)
                                     .apply(pd.Series.first_valid_index,axis=1)
                                     )
        return pd.DataFrame(triple_barrier)

    def _remove_double_barrier_crossings(self, triple_barrier: pd.DataFrame) -> pd.DataFrame:
        """
        Remove double barrier crossings from the barrier outcomes. This happens since to cross
        a 2nd (positive or negative) barrier, their respective 1st barrier needs to have been 
        crossed before.

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
        # .idxmin() returns np.nan if no barriers have been hit 
        # => .fillna(0) labels the "no barrier hit" condition
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
            self._labels = self._get_labels()
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
        if self._labels is None:
            raise AttributeError("The attribute 'labels' have not been calculated yet. Run .labels")
        transition_counts = self.labels.groupby([self.labels.rename("From"), self.labels.shift(-1).rename("To")]).size()

        transition_probas = transition_counts.groupby(level=0).apply(lambda x: x / x.sum())
        # Bring it into a proper form
        transition_probas = transition_probas.unstack().droplevel(1).fillna(0)
        transition_probas.index = transition_probas.index.astype(int)
        transition_probas.columns = transition_probas.columns.astype(int)
        return transition_probas

    def plot_at_date(self, date: str, months=3, figsize=(12,3)):
        date = pd.Timestamp(date)
        start_date = date - pd.DateOffset(months=months)
        end_date = date + pd.DateOffset(months=months)
        # Create data to plot
        index = np.cumprod(1 + self.returns.loc[start_date:end_date])
        if date not in index.index:
            warnings.warn_explicit(
                f"The selected date '{date.strftime('%Y-%m-%d')}' is not inside the attribute 'returns'." + 
                f"The next available date is '{index[date:].index.min().strftime('%Y-%m-%d')}'.",
                UserWarning, "", 1
            )

            date = index[date:].index.min()
        index = index / index.loc[date] * 100

        barrier_idx = index.loc[date:].head(self.n).index
        barriers = pd.concat([
            pd.Series(100*(1+2*self.barrier), index=barrier_idx).rename(2),
            pd.Series(100*(1+self.barrier), index=barrier_idx).rename(1),
            pd.Series(100*(1+-1*self.barrier), index=barrier_idx).rename(-1),
            pd.Series(100*(1+-2*self.barrier), index=barrier_idx).rename(-2),
        ], axis=1)

        # Plotting
        fig, ax = plt.subplots(figsize=figsize)

        aut_locator = mdates.AutoDateLocator(minticks=months*2)
        aut_formatter = mdates.ConciseDateFormatter(aut_locator)

        plt.axhline(100, c="grey", ls="--", lw=0.5)
        plt.title(f"Triple Barrier Method for {self.returns.name} at {date.strftime('%d %b, %Y')}")
        index.plot(ax=ax)
        barriers[2].plot(ax=ax, c="green")
        barriers[1].plot(ax=ax, c="lightgreen")
        plt.plot([barriers.index.min(), barriers.index.min()], [barriers[-2].min(), barriers[2].max()], ls="dotted", c="grey")
        plt.plot([barriers.index.max(), barriers.index.max()], [barriers[-2].min(), barriers[2].max()], label=0, c="black")
        barriers[-1].plot(ax=ax, c="lightsalmon")
        barriers[-2].plot(ax=ax, c="salmon")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
        ax.xaxis.set_major_locator(aut_locator)
        ax.xaxis.set_major_formatter(aut_formatter)
        ax.set_xlabel(None)
        plt.xlim(start_date, end_date)
        plt.show()
