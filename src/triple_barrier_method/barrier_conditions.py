
from dataclasses import dataclass, field
from typing import Dict, Callable

@dataclass
class BarrierConditions:
    n: int
    barrier: float
    conditions: Dict[int, Callable[[float], bool]] = field(default_factory=dict)

    def __post_init__(self):
        self.generate_conditions()
        self.sort_conditions()

    def generate_conditions(self):
        for i in range(1, self.n):
            self.conditions[-i] = self._generate_condition(
                lower_bound=(-i - 1) * self.barrier,
                upper_bound=-i * self.barrier
            )
            self.conditions[i] = self._generate_condition(
                lower_bound=i * self.barrier,
                upper_bound=(i + 1) * self.barrier
            )
        self.conditions[-self.n] = lambda x: x <= -self.n * self.barrier
        self.conditions[self.n] = lambda x: x > self.n * self.barrier

    def _generate_condition(self, lower_bound: float, upper_bound: float) -> Callable[[float], bool]:
        return lambda x: (x > lower_bound) & (x <= upper_bound)

    def sort_conditions(self):
        self.conditions = dict(sorted(self.conditions.items()))

# TODO: Tests
# vals = np.linspace(-4*barrier, 4*barrier, 9)
# pd.DataFrame({val: {k*barrier: v(val) for k,v in tbm.conditions.items()} for val in vals}).astype(int).replace(0, "")