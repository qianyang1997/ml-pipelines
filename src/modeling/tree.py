import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Union, List, Optional
from src.modeling.validation import Validation

from sklearn import datasets


def prepare_data(data: pd.DataFrame,
                 features: Union[str, List[str], np.ndarray],
                 response: str):
    
    X = None
    y = None
    
    # categorical variables
    
    return X, y
