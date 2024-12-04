from sklearn.metrics import brier_score_loss

# True binary labels
y_true = [0, 1, 1, 0]

# Predicted probabilities for the positive class
y_proba = [0.1, 0.9, 0.8, 0.4]

# Calculate Brier score loss
score = brier_score_loss(y_true, y_proba)

print(f"Brier Score Loss: {score}")