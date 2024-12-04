import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibrationDisplay

def from_estimator(estimator, X, y, n_bins=10, strategy='uniform', pos_label=None, name=None, ref_line=True, ax=None, **kwargs):
    """
    Plot a calibration curve using a binary classifier and data.

    Parameters:
    - estimator: Fitted classifier
    - X: Input values
    - y: Binary target values
    - n_bins: Number of bins for discretization
    - strategy: Bin width strategy ('uniform' or 'quantile')
    - pos_label: Positive class label
    - name: Label for the curve
    - ref_line: Whether to plot a reference line
    - ax: Matplotlib axes
    - **kwargs: Additional keyword arguments for plotting

    Returns:
    - CalibrationDisplay object that stores computed values
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Predict probabilities
    y_prob = estimator.predict_proba(X)[:, 1]

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=n_bins, strategy=strategy, pos_label=pos_label)

    # Plot calibration curve
    display = CalibrationDisplay(prob_true=prob_true, prob_pred=prob_pred, estimator_name=name)
    display.plot(ax=ax, **kwargs)

    # Plot reference line
    if ref_line:
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')

    ax.set_title('Calibration Curve')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')

    return display

