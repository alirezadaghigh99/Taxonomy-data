import torch

def _apply_categorical_projection(y, y_probs, z):
    # Ensure inputs are torch tensors
    y = torch.tensor(y, dtype=torch.float32)
    y_probs = torch.tensor(y_probs, dtype=torch.float32)
    z = torch.tensor(z, dtype=torch.float32)

    # Check the shapes of the inputs
    batch_size, n_atoms = y.shape
    assert y_probs.shape == (batch_size, n_atoms), "y_probs must have the same shape as y"
    assert z.shape == (n_atoms,), "z must have shape (n_atoms,)"

    # Calculate the spacing between z values
    delta_z = z[1] - z[0]

    # Initialize the output probabilities
    projected_probs = torch.zeros_like(y_probs)

    # Iterate over each batch
    for b in range(batch_size):
        # Clamp y values to be within the range of z
        y_clamped = torch.clamp(y[b], min=z[0], max=z[-1])

        # Calculate bj
        bj = (y_clamped - z[0]) / delta_z

        # Calculate floor and ceil of bj
        l = torch.floor(bj).long()
        u = torch.ceil(bj).long()

        # Accumulate m_l and m_u
        m_l = (u.float() - bj) * y_probs[b]
        m_u = (bj - l.float()) * y_probs[b]

        # Distribute probabilities to the lower and upper indices
        for j in range(n_atoms):
            if l[j] >= 0 and l[j] < n_atoms:
                projected_probs[b, l[j]] += m_l[j]
            if u[j] >= 0 and u[j] < n_atoms:
                projected_probs[b, u[j]] += m_u[j]

    return projected_probs.numpy()

