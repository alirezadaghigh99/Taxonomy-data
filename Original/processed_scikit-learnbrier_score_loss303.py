def brier_score_loss(
    y_true, y_proba=None, *, sample_weight=None, pos_label=None, y_prob="deprecated"
):
    """Compute the Brier score loss.

    The smaller the Brier score loss, the better, hence the naming with "loss".
    The Brier score measures the mean squared difference between the predicted
    probability and the actual outcome. The Brier score always
    takes on a value between zero and one, since this is the largest
    possible difference between a predicted probability (which must be
    between zero and one) and the actual outcome (which can take on values
    of only 0 and 1). It can be decomposed as the sum of refinement loss and
    calibration loss.

    The Brier score is appropriate for binary and categorical outcomes that
    can be structured as true or false, but is inappropriate for ordinal
    variables which can take on three or more values (this is because the
    Brier score assumes that all possible outcomes are equivalently
    "distant" from one another). Which label is considered to be the positive
    label is controlled via the parameter `pos_label`, which defaults to
    the greater label unless `y_true` is all 0 or all -1, in which case
    `pos_label` defaults to 1.

    Read more in the :ref:`User Guide <brier_score_loss>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.

    y_proba : array-like of shape (n_samples,)
        Probabilities of the positive class.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    pos_label : int, float, bool or str, default=None
        Label of the positive class. `pos_label` will be inferred in the
        following manner:

        * if `y_true` in {-1, 1} or {0, 1}, `pos_label` defaults to 1;
        * else if `y_true` contains string, an error will be raised and
          `pos_label` should be explicitly specified;
        * otherwise, `pos_label` defaults to the greater label,
          i.e. `np.unique(y_true)[-1]`.

    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.

        .. deprecated:: 1.5
            `y_prob` is deprecated and will be removed in 1.7. Use
            `y_proba` instead.

    Returns
    -------
    score : float
        Brier score loss.

    References
    ----------
    .. [1] `Wikipedia entry for the Brier score
            <https://en.wikipedia.org/wiki/Brier_score>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import brier_score_loss
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_true_categorical = np.array(["spam", "ham", "ham", "spam"])
    >>> y_prob = np.array([0.1, 0.9, 0.8, 0.3])
    >>> brier_score_loss(y_true, y_prob)
    np.float64(0.037...)
    >>> brier_score_loss(y_true, 1-y_prob, pos_label=0)
    np.float64(0.037...)
    >>> brier_score_loss(y_true_categorical, y_prob, pos_label="ham")
    np.float64(0.037...)
    >>> brier_score_loss(y_true, np.array(y_prob) > 0.5)
    np.float64(0.0)
    """
    # TODO(1.7): remove in 1.7 and reset y_proba to be required
    # Note: validate params will raise an error if y_prob is not array-like,
    # or "deprecated"
    if y_proba is not None and not isinstance(y_prob, str):
        raise ValueError(
            "`y_prob` and `y_proba` cannot be both specified. Please use `y_proba` only"
            " as `y_prob` is deprecated in v1.5 and will be removed in v1.7."
        )
    if y_proba is None:
        warnings.warn(
            (
                "y_prob was deprecated in version 1.5 and will be removed in 1.7."
                "Please use ``y_proba`` instead."
            ),
            FutureWarning,
        )
        y_proba = y_prob

    y_true = column_or_1d(y_true)
    y_proba = column_or_1d(y_proba)
    assert_all_finite(y_true)
    assert_all_finite(y_proba)
    check_consistent_length(y_true, y_proba, sample_weight)

    y_type = type_of_target(y_true, input_name="y_true")
    if y_type != "binary":
        raise ValueError(
            "Only binary classification is supported. The type of the target "
            f"is {y_type}."
        )

    if y_proba.max() > 1:
        raise ValueError("y_proba contains values greater than 1.")
    if y_proba.min() < 0:
        raise ValueError("y_proba contains values less than 0.")

    try:
        pos_label = _check_pos_label_consistency(pos_label, y_true)
    except ValueError:
        classes = np.unique(y_true)
        if classes.dtype.kind not in ("O", "U", "S"):
            # for backward compatibility, if classes are not string then
            # `pos_label` will correspond to the greater label
            pos_label = classes[-1]
        else:
            raise
    y_true = np.array(y_true == pos_label, int)
    return np.average((y_true - y_proba) ** 2, weights=sample_weight)