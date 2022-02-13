import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Union, List, Optional
from src.modeling.train_val_test import Pipeline

from sklearn import datasets


def prepare_data(data: pd.DataFrame,
                 features: Union[str, List[str], np.ndarray],
                 response: str):
    
    X = None
    y = None
    
    # categorical variables
    
    return X, y

def pipeline(X, y, X_test, y_test, **kwargs):
    
    obj = Pipeline(X, y , X_test, y_test,
                     estimator=xgb.XGBRFRegressor,
                     model_name='xgboost',
                     **kwargs)
    
    #results, best = obj.tune_cv() # specify n_iter
    #obj.save_model() # specify version or filepath
    #obj.cv_eval()
    #obj.fit_test(scores=['mse', 'r2'])
    
    # or...
    #obj.fit(custom_X, custom_y)
    #obj.model.predict(custom_y_test)
    
    # or...
    
    