import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Union, List, Optional

from config.config import PARAMS

from sklearn import datasets


def prepare_data(data: pd.DataFrame,
                 features: Union[str, List[str], np.ndarray],
                 response: str):
    
    X = None
    y = None
    
    # categorical variables
    
    return X, y
    
def fit(X, y,
        validation: bool=True,
        ):
    

iris = datasets.load_iris()