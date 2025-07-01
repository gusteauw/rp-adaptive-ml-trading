# =============================================
# Module: walkforward_cv.py
# Utility for Walk-Forward Cross-Validation
# =============================================

import numpy as np

def walk_forward_split(X, n_splits=5, test_size=0.2, min_train_size=None):
    """
    Walk-forward expanding window cross-validation.
    
    Parameters:
    - X (array-like or DataFrame): Dataset to split (only needs shape).
    - n_splits (int): Number of walk-forward iterations.
    - test_size (float or int): Size of test set (fraction or fixed count).
    - min_train_size (int): Minimum size of the training set (optional).
    
    Yields:
    - (train_idx, test_idx): indices for training and testing sets.
    """
    n_samples = len(X)
    
    if isinstance(test_size, float):
        test_size = int(test_size * n_samples)
    if test_size <= 0 or test_size >= n_samples:
        raise ValueError("Invalid test_size")

    split_step = (n_samples - test_size) // n_splits
    for i in range(n_splits):
        train_end = split_step * (i + 1)
        test_start = train_end
        test_end = test_start + test_size

        if test_end > n_samples:
            break

        train_start = 0 if min_train_size is None else max(0, test_start - min_train_size)
        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)

        yield train_idx, test_idx
