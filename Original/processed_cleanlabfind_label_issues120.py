def find_label_issues(
    labels: list,
    pred_probs: np.ndarray,
    return_indices_ranked_by: Optional[str] = None,
    rank_by_kwargs={},
    filter_by: str = "prune_by_noise_rate",
    frac_noise: float = 1.0,
    num_to_remove_per_class: Optional[List[int]] = None,
    min_examples_per_class=1,
    confident_joint: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    low_memory: bool = False,
) -> np.ndarray:
    """
    Identifies potentially mislabeled examples in a multi-label classification dataset.
    An example is flagged as with a label issue if *any* of the classes appear to be incorrectly annotated for this example.

    Parameters
    ----------
    labels : List[List[int]]
      List of noisy labels for multi-label classification where each example can belong to multiple classes.
      This is an iterable of iterables where the i-th element of `labels` corresponds to a list of classes that the i-th example belongs to,
      according to the original data annotation (e.g. ``labels = [[1,2],[1],[0],..]``).
      This method will return the indices i where the inner list ``labels[i]`` is estimated to have some error.
      For a dataset with K classes, each class must be represented as an integer in 0, 1, ..., K-1 within the labels.

    pred_probs : np.ndarray
      An array of shape ``(N, K)`` of model-predicted class probabilities.
      Each row of this matrix corresponds to an example `x`
      and contains the predicted probability that `x` belongs to each possible class,
      for each of the K classes (along its columns).
      The columns need not sum to 1 but must be ordered such that
      these probabilities correspond to class 0, 1, ..., K-1.

      Note
      ----
      Estimated label quality scores are most accurate when they are computed based on out-of-sample ``pred_probs`` from your model.
      To obtain out-of-sample predicted probabilities for every example in your dataset, you can use :ref:`cross-validation <pred_probs_cross_val>`.
      This is encouraged to get better results.

    return_indices_ranked_by : {None, 'self_confidence', 'normalized_margin', 'confidence_weighted_entropy'}, default = None
      This function can return a boolean mask (if None) or an array of the example-indices with issues sorted based on the specified ranking method.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    rank_by_kwargs : dict, optional
      Optional keyword arguments to pass into scoring functions for ranking by
      label quality score (see :py:func:`rank.get_label_quality_scores
      <cleanlab.rank.get_label_quality_scores>`).

    filter_by : {'prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given', 'low_normalized_margin', 'low_self_confidence'}, default='prune_by_noise_rate'
      The specific Confident Learning method to determine precisely which examples have label issues in a dataset.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    frac_noise : float, default = 1.0
      This will return the "top" frac_noise * num_label_issues estimated label errors, dependent on the filtering method used,
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    num_to_remove_per_class : array_like
      An iterable that specifies the number of mislabeled examples to return from each class.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    min_examples_per_class : int, default = 1
      The minimum number of examples required per class below which examples from this class will not be flagged as label issues.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    confident_joint : np.ndarray, optional
      An array of shape ``(K, 2, 2)`` representing a one-vs-rest formatted confident joint, as is appropriate for multi-label classification tasks.
      Entry ``(c, i, j)`` in this array is the number of examples confidently counted into a ``(class c, noisy label=i, true label=j)`` bin,
      where `i, j` are either 0 or 1 to denote whether this example belongs to class `c` or not
      (recall examples can belong to multiple classes in multi-label classification).
      The `confident_joint` can be computed using :py:func:`count.compute_confident_joint <cleanlab.count.compute_confident_joint>` with ``multi_label=True``.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    n_jobs : optional
      Number of processing threads used by multiprocessing.
      Refer to documentation for this argument in :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` for details.

    verbose : optional
      If ``True``, prints when multiprocessing happens.

    low_memory: bool, default=False
      Set as ``True`` if you have a big dataset with limited memory.
      Uses :py:func:`experimental.label_issues_batched.find_label_issues_batched <cleanlab.experimental.label_issues_batched>`

    Returns
    -------
    label_issues : np.ndarray
      If `return_indices_ranked_by` left unspecified, returns a boolean **mask** for the entire dataset
      where ``True`` represents an example suffering from some label issue and
      ``False`` represents an example that appears accurately labeled.

      If `return_indices_ranked_by` is specified, this method instead returns a list of **indices** of examples identified with
      label issues (i.e. those indices where the mask would be ``True``).
      Indices are sorted by the likelihood that *all* classes are correctly annotated for the corresponding example.

      Note
      ----
      Obtain the *indices* of examples with label issues in your dataset by setting
      `return_indices_ranked_by`.

    """
    from cleanlab.filter import _find_label_issues_multilabel

    if low_memory:
        if rank_by_kwargs:
            warnings.warn(f"`rank_by_kwargs` is not used when `low_memory=True`.")

        func_signature = inspect.signature(find_label_issues)
        default_args = {
            k: v.default
            for k, v in func_signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        arg_values = {
            "filter_by": filter_by,
            "num_to_remove_per_class": num_to_remove_per_class,
            "confident_joint": confident_joint,
            "n_jobs": n_jobs,
            "num_to_remove_per_class": num_to_remove_per_class,
            "frac_noise": frac_noise,
            "min_examples_per_class": min_examples_per_class,
        }
        for arg_name, arg_val in arg_values.items():
            if arg_val != default_args[arg_name]:
                warnings.warn(f"`{arg_name}` is not used when `low_memory=True`.")

    return _find_label_issues_multilabel(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by=return_indices_ranked_by,
        rank_by_kwargs=rank_by_kwargs,
        filter_by=filter_by,
        frac_noise=frac_noise,
        num_to_remove_per_class=num_to_remove_per_class,
        min_examples_per_class=min_examples_per_class,
        confident_joint=confident_joint,
        n_jobs=n_jobs,
        verbose=verbose,
        low_memory=low_memory,
    )