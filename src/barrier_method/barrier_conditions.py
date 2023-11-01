
from dataclasses import dataclass, field
from typing import Dict, Callable

@dataclass
class BarrierConditions:
    """
    A class that generates and manages barrier conditions used for different labeling techniques.

    Attributes:
    n (int): The number of barrier conditions to be generated.
    barrier (float): The threshold value for the barrier.
    conditions (Dict[int, Callable[[float], bool]]): A dictionary holding condition functions for various barrier levels.
    """
    n: int
    barrier: float
    conditions: Dict[int, Callable[[float], bool]] = field(default_factory=dict)

    def __post_init__(self):
        """
        Calculate the conditions after the instance has been initialized.
        """
        self.generate_conditions()
        self.sort_conditions()

    def generate_conditions(self):
        """
        Generates barrier conditions based on the specified number of conditions and threshold values.
        """
        for i in range(1, self.n):
            # Negative barrier labels, e.g. return between -20% and -10%
            self.conditions[-i] = self._generate_condition(
                lower_bound=(-i - 1) * self.barrier,
                upper_bound=-i * self.barrier
            )
            # Positive barrier labels, e.g. return between 10% and 20%
            self.conditions[i] = self._generate_condition(
                lower_bound=i * self.barrier,
                upper_bound=(i + 1) * self.barrier
            )
        # Set conditions for the extreme values, e.g.:
        # return > 20% for the extreme positive case
        # return <= -20% for the extreme negative case
        self.conditions[-self.n] = lambda x: x <= -self.n * self.barrier
        self.conditions[self.n] = lambda x: x > self.n * self.barrier

    def _generate_condition(self, lower_bound: float, upper_bound: float) -> Callable[[float], bool]:
        """
        Generates a condition function based on the given lower and upper bounds.

        Args:
        lower_bound (float): Lower threshold for the condition.
        upper_bound (float): Upper threshold for the condition.

        Returns:
        Callable[[float], bool]: Condition function that checks if the input value satisfies the given bounds.
        """
        return lambda x: (x > lower_bound) & (x <= upper_bound)

    def sort_conditions(self):
        """
        Sorts the generated conditions in ascending order based on their keys.
        """
        self.conditions = dict(sorted(self.conditions.items()))

# TODO: Tests
# vals = np.linspace(-4*barrier, 4*barrier, 9)
# pd.DataFrame({val: {k*barrier: v(val) for k,v in tbm.conditions.items()} for val in vals}).astype(int).replace(0, "")