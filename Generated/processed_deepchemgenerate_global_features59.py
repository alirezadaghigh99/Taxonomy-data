from rdkit import Chem
import numpy as np

# Assuming GraphConvConstants is defined somewhere with FEATURE_GENERATORS
class GraphConvConstants:
    FEATURE_GENERATORS = {
        'example_generator': lambda mol: [1.0, 2.0, 3.0]  