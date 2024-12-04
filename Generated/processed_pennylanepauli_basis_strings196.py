from itertools import product

def pauli_basis_strings(num_wires):
    """
    Generate all n-qubit Pauli words except the identity string "I"*num_wires.

    Args:
    num_wires (int): Number of qubits (wires).

    Returns:
    List[str]: List of Pauli words in lexicographical order.
    """
    pauli_letters = ['I', 'X', 'Y', 'Z']
    all_pauli_words = [''.join(word) for word in product(pauli_letters, repeat=num_wires)]
    
    # Remove the identity string "I"*num_wires
    identity_string = 'I' * num_wires
    pauli_words = [word for word in all_pauli_words if word != identity_string]
    
    return pauli_words

