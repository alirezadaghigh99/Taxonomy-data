import torch

class ParametrizedLine:
    def __init__(self, p0, direction):
        self.p0 = p0
        self.direction = direction

    @staticmethod
    def through(p0, p1):
        # Calculate the direction vector from p0 to p1
        direction = p1 - p0
        
        # Calculate the norm of the direction vector
        norm = torch.norm(direction, dim=1, keepdim=True)
        
        # Normalize the direction vector
        normalized_direction = direction / norm
        
        # Return an instance of ParametrizedLine
        return ParametrizedLine(p0, normalized_direction)

