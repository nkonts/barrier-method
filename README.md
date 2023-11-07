
The repository expands the `Triple-Barrier Method` proposed by Marcos López de Prado. It introduces specific enhancements:

- **Extension to Multiple Barrier Conditions**: Dynamically generates multiple barriers through the `n_barriers` parameter.
- **Centering Capability**: Includes a `center` parameter to center the returns for improved analysis.
- **Improved Readability**: Utilizes standard data science toolkit with `pandas` and `numpy` instead of for-loops for each time-step `t` to improve readability and efficiency.
- **Exporting Intermediate Steps**: Allows to view intermediate steps as features for model usage (`_check_barrier_crossing()` method).
- **Convenient Properties**: Provides useful properties such as `transition_probabilities` and `signals_pa`.


# Notebooks:

This repository includes two notebooks:

1) **`example.ipynb`**:  Provides an example showcasing the usage of the `BarrierMethod` class.
2) **`study.ipynb`**:  Generates various labels for a range of n and different barriers and tests them for a normal and uniform distribution.

    For some use cases, the goal might be to have (on average) a new signal each week that is approx. normal distributed (neutral most of the time, rare tails) or uniformly distributed.
    

# Examples:
![Barrier Method](figures/barrier_method_example.png "Barrier Method")
![Barrier Frequency](figures/barrier_freq.png "Barrier Frequency")
![Barrier Frequency Centered](figures/barrier_freq_centered.png "Barrier Frequency Centered")


# Triple-Barrier Method

The traditional approach to labeling data involves a fixed-time horizon. However, Marcos López de Prado introduced the `Triple-Barrier Method` in his book *Advances in Financial Machine Learning* (Wiley, 2018). This method constructs three barriers for each observation:

- **Stop-loss Barrier (`-1`)**: Indicates a selling opportunity.
- **Neutral Signal (`0`)**: Denotes a scenario where the time series doesn't cross any barrier within a maximum holding period of `n`.
- **Profit-taking Barrier (`+1`)**: Signifies a buying opportunity.


### A visual example:
![Triple-Barrier Method](references/triple_barrier.png "Triple-Barrier Method")


# Installation:

```bash
pip install git+https://github.com/nkonts/barrier-method.git
