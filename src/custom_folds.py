import math
from datetime import datetime
import pandas as pd

import logging

logger = logging.getLogger(__name__)


def time_series_split(n_fold, start_ym, end_ym):
    """divide dataset into CV folds based on year-month; if n_fold does not divide
    the number of months in a designated range, the leftover extra fold would be the final validation set
    (in other words, the dataset would have n_fold + 1 folds)"""
    # calculate the number of year-months between start_ym and end_ym, inclusive
    pr = pd.period_range(start=start_ym, end=end_ym, freq='M')
    pr = [str(period) for period in pr]
    number_of_ym = len(pr)
    assert len(pr) >= 2, 2 <= n_fold <= len(pr)
    ym_in_fold = number_of_ym // n_fold
    leftover = number_of_ym % n_fold
    # identify folds - each fold has train_X, train_y, test_x, test_y
    folds = dict()
    i = 0
    for j in range(0, len(pr) - leftover, ym_in_fold):
        folds[i] = [pr[j + k] for k in range(ym_in_fold)]
        i += 1
    if leftover != 0:
        folds[i] = pr[-leftover:]
    # add logging statements
    logger.debug(f'The folds are: {folds}.')
    return folds


def time_folds(n_fold, start_ym, end_ym, steps=1):
    """
    Function that creates mini step train / validation splits to maximize the amount of data inside each fold
    Args:
        n_fold: (int) indicating the number of folds to make
        start_ym: string that determines the start year month of the folds (included)
        end_ym: string that determines the end year month of the folds (included)
        steps: (int) indicating the number of steps to take per train/validation
    Returns:
        Dictionary of lists. Each list contains 4 dates -> train start, train end, validation start, validation end
    """
    # if not string then change to string (assuming it is pandas period)
    start_ym = str(start_ym) if not isinstance(start_ym, str) else start_ym
    end_ym = str(end_ym) if not isinstance(end_ym, str) else end_ym

    start = datetime.strptime(start_ym, '%Y-%m')
    end = datetime.strptime(end_ym, '%Y-%m')

    mrange = (end.year - start.year) * 12 + (end.month - start.month)
    halfrg = math.ceil((mrange - n_fold * steps) / 2) + 1

    tstyear, tstmonth = start.year, start.month
    tedyear = start.year + int(halfrg / 12)
    tedmonth = start.month + halfrg - int(halfrg / 12) * 12

    vstyear, vstmonth = (start.year + int(halfrg / 12), start.month + halfrg - int(halfrg / 12) * 12 + 1) \
        if start.month + halfrg - int(halfrg / 12) * 12 + 1 <= 12 \
        else (start.year + int(halfrg / 12) + 1, start.month + halfrg - int(halfrg / 12) * 12 - 11)
    vedyear, vedmonth = (end.year, end.month - n_fold * steps + 1) \
        if end.month - n_fold * steps + 1 >= 1 \
        else (end.year - 1, end.month - n_fold * steps + 13)
    bymonth = dict()
    for n, i in enumerate([0] + [steps] * (n_fold - 1)):
        tstyear, tstmonth = (tstyear, tstmonth + i) if tstmonth + i <= 12 else (tstyear + 1, tstmonth + i - 12)
        tedyear, tedmonth = (tedyear, tedmonth + i) if tedmonth + i <= 12 else (tedyear + 1, tedmonth + i - 12)
        vstyear, vstmonth = (vstyear, vstmonth + i) if vstmonth + i <= 12 else (vstyear + 1, vstmonth + i - 12)
        vedyear, vedmonth = (vedyear, vedmonth + i) if vedmonth + i <= 12 else (vedyear + 1, vedmonth + i - 12)
        bymonth[n] = [datetime(tstyear, tstmonth, 1).strftime('%Y-%m'),
                      datetime(tedyear, tedmonth, 1).strftime('%Y-%m'),
                      datetime(vstyear, vstmonth, 1).strftime('%Y-%m'),
                      datetime(vedyear, vedmonth, 1).strftime('%Y-%m')]

    # add logging statements
    logger.debug(f'The splits are: {bymonth}.')

    return bymonth