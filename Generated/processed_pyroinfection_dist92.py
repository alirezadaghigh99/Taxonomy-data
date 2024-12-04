import numpy as np
from scipy.stats import poisson, nbinom, binom, betabinom

def infection_dist(individual_rate, num_infectious, num_susceptible, population, concentration, overdispersion):
    """
    Creates a distribution over the number of new infections at a discrete time step.
    
    Parameters:
    - individual_rate: The rate at which an individual can infect others.
    - num_infectious: The number of currently infectious individuals.
    - num_susceptible: The number of susceptible individuals.
    - population: The total population size.
    - concentration: A parameter that affects the distribution choice.
    - overdispersion: A parameter that models the variability in the infection process.
    
    Returns:
    - A distribution object (Poisson, Negative-Binomial, Binomial, or Beta-Binomial).
    """
    
    # Calculate the basic reproduction number (R0)
    R0 = individual_rate * num_infectious
    
    # Calculate the expected number of new infections
    expected_infections = R0 * (num_susceptible / population)
    
    if population > 1000 and concentration > 1:
        # Large population and high concentration: Poisson distribution
        return poisson(mu=expected_infections)
    
    elif population > 1000 and concentration <= 1:
        # Large population and low concentration: Negative-Binomial distribution
        # Negative-Binomial parameters
        p = 1 / (1 + expected_infections / overdispersion)
        r = overdispersion
        return nbinom(n=r, p=p)
    
    elif population <= 1000 and concentration > 1:
        # Small population and high concentration: Binomial distribution
        # Binomial parameters
        n = num_susceptible
        p = expected_infections / num_susceptible
        return binom(n=n, p=p)
    
    else:
        # Small population and low concentration: Beta-Binomial distribution
        # Beta-Binomial parameters
        alpha = expected_infections
        beta = num_susceptible - expected_infections
        n = num_susceptible
        return betabinom(n=n, a=alpha, b=beta)

