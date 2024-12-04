import numpy as np
import itertools

def _iterate_shift_rule_with_multipliers(rule, order, period=None):
    """
    Apply a shift rule with multipliers repeatedly along the same parameter axis.

    Args:
        rule (np.ndarray): A 2D array where each row contains [coefficient, multiplier, shift].
        order (int): The number of times to repeat the shift rule.
        period (float, optional): The period for handling periodic boundary conditions.

    Returns:
        np.ndarray: A stacked array of combined rules with columns corresponding to
                    coefficients, multipliers, and cumulative shifts.
    """
    # Extract coefficients, multipliers, and shifts from the rule
    coefficients = rule[:, 0]
    multipliers = rule[:, 1]
    shifts = rule[:, 2]

    # Generate all possible combinations of applying the rule `order` times
    combinations = list(itertools.product(range(len(rule)), repeat=order))

    combined_rules = []

    for combo in combinations:
        # Initialize cumulative values
        cumulative_coeff = 1.0
        cumulative_mult = 1.0
        cumulative_shift = 0.0

        for idx in combo:
            cumulative_coeff *= coefficients[idx]
            cumulative_mult *= multipliers[idx]
            cumulative_shift += shifts[idx]

        # Apply periodic boundary conditions if period is specified
        if period is not None:
            cumulative_shift = cumulative_shift % period

        combined_rules.append([cumulative_coeff, cumulative_mult, cumulative_shift])

    return np.array(combined_rules)

