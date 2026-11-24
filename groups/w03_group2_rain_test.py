"""Group 2 (w03) - Rain test: compare sales on rainy and dry days.

Authors: Yevhenii, Dzmitry, Darya
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ml_base import Analysis


class RainTest(Analysis):    # <- rename to your card's class, e.g. RainTest
    """Compare average sales on rainy and dry days with a two-sided shuffle test.

    Usage:
        model = RainTest(data).fit()
        model.t_test()
    """
    required_columns = ["revenue", "rain_mm"]

    def __init__(self, daily, threshold=1.0, n_shuffles=1000, seed=0):
        """Set the rain cutoff in mm and the number and seed of the shuffles.

        Raises ValueError for a negative cutoff or a shuffle count below one.
        """
        super().__init__(daily)
        if threshold < 0:
            raise ValueError("The rain threshold must be at least 0 mm.")
        if n_shuffles < 1:
            raise ValueError("Use at least one shuffle.")
        self.threshold = threshold
        self.n_shuffles = n_shuffles
        self.seed = seed

    def fit(self):
        """Store the day counts, revenue differences and two-sided p-value.

        Returns self. Raises ValueError when either group is empty.
        """
        self.results_ = None
        values = self.data["revenue"].to_numpy()
        rainy = self.data["rain_mm"].to_numpy() >= self.threshold
        if not rainy.any() or rainy.all():
            raise ValueError("Choose a threshold with both rainy and dry days.")

        rainy_mean = values[rainy].mean()
        dry_mean = values[~rainy].mean()
        difference = rainy_mean - dry_mean
        rng = np.random.default_rng(self.seed)
        self.shuffled_ = np.empty(self.n_shuffles)
        for i in range(self.n_shuffles):
            labels = rng.permutation(rainy)
            self.shuffled_[i] = values[labels].mean() - values[~labels].mean()

        self.results_ = {
            "rainy_days": int(rainy.sum()),
            "dry_days": int((~rainy).sum()),
            "difference": float(difference),
            "difference_pct": float(100 * difference / dry_mean),
            "p_value": float(np.mean(np.abs(self.shuffled_) >= abs(difference))),
        }
        return self

    def t_test(self):
        """Return Welch's test p-value to three decimals. Fit the model first.

        Raises ValueError if either group has fewer than two days.
        """
        self.check_fitted()
        rainy = self.data["rain_mm"] >= self.threshold
        if rainy.sum() < 2 or (~rainy).sum() < 2:
            raise ValueError("Welch's test needs at least two days in each group.")
        revenue = self.data["revenue"]
        result = stats.ttest_ind(revenue[rainy], revenue[~rainy], equal_var=False)
        return round(float(result.pvalue), 3)

    def is_significant(self, alpha=0.05):
        """Return whether the shuffle p-value is below alpha. Fit the model first."""
        self.check_fitted()
        return self.results_["p_value"] < alpha

    def plot(self):
        """Return the histogram Axes with a line at the observed difference."""
        self.check_fitted()
        fig, ax = plt.subplots()
        ax.hist(self.shuffled_, bins=30)
        ax.axvline(self.results_["difference"], color="red", label="Observed difference")
        ax.set(xlabel="Difference in average revenue (GBP)", ylabel="Number of shuffles")
        ax.legend()
        return ax
