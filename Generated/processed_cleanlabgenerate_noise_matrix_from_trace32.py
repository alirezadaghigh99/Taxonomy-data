import numpy as np

def generate_noise_matrix_from_trace(K, trace, max_trace_prob, min_trace_prob, max_noise_rate, min_noise_rate, 
                                     valid_noise_matrix=True, py=None, frac_zero_noise_rates=0.0, seed=None, max_iter=10000):
    if seed is not None:
        np.random.seed(seed)
    
    if valid_noise_matrix and (py is None or len(py) != K):
        raise ValueError("py must be provided and have length K when valid_noise_matrix is True.")
    
    for _ in range(max_iter):
        noise_matrix = np.zeros((K, K))
        
        # Generate diagonal elements
        diagonal_elements = np.random.uniform(min_trace_prob, max_trace_prob, K)
        diagonal_elements = diagonal_elements / np.sum(diagonal_elements) * trace
        
        if np.any(diagonal_elements > 1):
            continue
        
        np.fill_diagonal(noise_matrix, diagonal_elements)
        
        # Generate off-diagonal elements
        for i in range(K):
            remaining_prob = 1 - noise_matrix[i, i]
            off_diagonal_elements = np.random.uniform(min_noise_rate, max_noise_rate, K-1)
            off_diagonal_elements = off_diagonal_elements / np.sum(off_diagonal_elements) * remaining_prob
            
            if np.any(off_diagonal_elements > 1):
                break
            
            noise_matrix[i, np.arange(K) != i] = off_diagonal_elements
        
        # Check if the matrix is valid
        if valid_noise_matrix:
            valid = True
            for i in range(K):
                for j in range(K):
                    if i != j and py[i] * noise_matrix[i, j] >= noise_matrix[i, i]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
        
        # Apply frac_zero_noise_rates
        if frac_zero_noise_rates > 0:
            num_zero_elements = int(frac_zero_noise_rates * K * (K - 1))
            zero_indices = np.random.choice(K * (K - 1), num_zero_elements, replace=False)
            for idx in zero_indices:
                i, j = divmod(idx, K - 1)
                if j >= i:
                    j += 1
                noise_matrix[i, j] = 0
        
        return noise_matrix
    
    return None

