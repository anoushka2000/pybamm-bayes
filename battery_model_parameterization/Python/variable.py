class Variable:
    """
    Parameter whose value is to be identified.

    Parameters
    ----------
    name: str
        Variable name
    true_value: float
        Value of variable used to simulate data.
    prior : pints.LogPrior
        Prior from which variable is sampled.
    """

    def __init__(self, name, true_value, prior):
        self.name = name
        self.true_value = true_value
        self.prior = prior
        self.prior_type = str(type(self.prior)).split(".")[-1][:-2]
