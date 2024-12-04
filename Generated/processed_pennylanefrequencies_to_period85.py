import math
from functools import reduce
from fractions import Fraction

def gcd(a, b):
    """Compute the greatest common divisor of a and b."""
    while b:
        a, b = b, a % b
    return a

def gcd_multiple(numbers):
    """Compute the GCD of a list of numbers."""
    return reduce(gcd, numbers)

def frequencies_to_period(frequencies, decimals=None):
    """
    Calculate the period of a Fourier series based on a set of frequencies.
    
    Parameters:
    frequencies (tuple): A tuple of frequencies.
    decimals (int, optional): Number of decimal places to round to.
    
    Returns:
    float: The period of the Fourier series.
    
    Example:
    frequencies = (0.5, 1.0)
    frequencies_to_period(frequencies)
    # Expected output: 12.566370614359172
    """
    if decimals is not None:
        frequencies = tuple(round(f, decimals) for f in frequencies)
    
    # Convert frequencies to fractions to handle non-integer values
    fractions = [Fraction(f).limit_denominator() for f in frequencies]
    
    # Find the least common multiple (LCM) of the denominators
    denominators = [frac.denominator for frac in fractions]
    lcm_denominator = reduce(lambda a, b: a * b // gcd(a, b), denominators)
    
    # Find the GCD of the numerators
    numerators = [frac.numerator * (lcm_denominator // frac.denominator) for frac in fractions]
    gcd_numerators = gcd_multiple(numerators)
    
    # The GCD of the frequencies is the GCD of the numerators divided by the LCM of the denominators
    gcd_frequencies = gcd_numerators / lcm_denominator
    
    # Calculate the period
    period = 2 * math.pi / gcd_frequencies
    
    return period

