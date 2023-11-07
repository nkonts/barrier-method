from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict


@dataclass
class Condition(ABC):
    factor: float
    barrier: float

    @abstractmethod
    def evaluate(self, value: float) -> bool:
        pass


@dataclass
class PositiveCondition(Condition):
    def evaluate(self, value: float) -> bool:
        return value > self.factor * self.barrier


@dataclass
class NegativeCondition(Condition):
    def evaluate(self, value: float) -> bool:
        return value < -1 * self.factor * self.barrier


@dataclass
class BarrierConditions:
    """
    A class that generates and manages barrier conditions used for different labeling techniques.
    Those conditions can be used to generate labels like this. Example for n=1:
        y =
            -1 if r_{t,t+n} < -barrier,
             1 if  r_{t,t+n} > -barrier,
             0 else
    Different n will add conditions by multiples of barrier up to n-multiples.

    Attributes:
    n (int): The number of barrier conditions to be generated for negative and positive barriers.
    barrier (float): The threshold value for the barrier.
    conditions (Dict[int, Condition): A dictionary holding condition functions for various barrier levels.
        Keys are sorted numerically.
    """
    n: int
    barrier: float
    conditions: Dict[int, Condition] = field(default_factory=dict)

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
        for i in range(1, self.n + 1):
            self.conditions[-i] = NegativeCondition(factor=i, barrier=self.barrier)
            self.conditions[i] = PositiveCondition(factor=i, barrier=self.barrier)

    def sort_conditions(self):
        """
        Sorts the generated conditions in ascending order based on their keys.
        """
        self.conditions = dict(sorted(self.conditions.items()))

    def __str__(self):
        """
        String representation of the BarrierConditions object, showing conditions in a readable format.
        """
        condition_strings = []
        for key, condition in self.conditions.items():
            condition_strings.append(f"\t{key}: \t{condition}")

        conditions_str = "\n\t".join(condition_strings)
        return f"BarrierConditions(n={self.n}, barrier={self.barrier}):\n\tConditions={{\n\t{conditions_str}\n\t}}"
