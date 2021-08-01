# Cross Validation with Time Series

For cross validation on time series data, the current sklearn package supports rudimentary train-validation splits if the datetime column is unique. But what if it's not?

Given time series data with non-unique datetime index, the pipeline in this repo performs cross validation with three methods:
1. The **sliding window** approach - train on period 1 data and validate on period 2, then train on period 2 data and validate on period 3, etc. Period 1, 2, ... are disjoint and adjacent to each other.
2. The **expanding window** approach - train on period 1 data and validate on period 2, then train on period 1 & 2 and validate on period 3, then train on period 1 & 2 & 3 and validate on period 4, etc. Period 1, 2, ... are disjoint and adjacent to each other.
3. The **step-by-step** approach - train on period 1 and validate on period n (with n being somewhere down the line), then train on period 2 and validate on period n + 1, etc. Period 1, 2, ... may overlap, and period n, n + 1... may overlap (but there is no overlap betwen 1, 2, ... and n, n + 1, ...).

The modules in this repo are designed to work with sklearn or sklearn-compatible estimators. There are examples of usage in the main section of `time_series_cv.py`.
