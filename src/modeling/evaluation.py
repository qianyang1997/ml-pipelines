import pandas as pd
from sklearn.utils.validation import check_is_fitted

from src.modeling.modeling import Modeling
from config.config import VAL_PARAMS
from utils.shap import create_shap_table

class Evaluation(Modeling):
    
    def __init__(self, model_name: str='xgboost', goal: str='regression',
                 load_version=None, load_filepath=None, fitted_model=None):
        
        self.model_name = model_name
        self.goal = goal
        if fitted_model is None:
            assert load_version or load_filepath
            self.model = self.load_model(version=load_version, filepath=load_filepath)
        else:
            self.model = fitted_model
        check_is_fitted(self.model)
        
    def sklearn_result(self, eval_func, X, y, refit=True):
        
        eval_params = VAL_PARAMS.get('eval_params', {})
        if refit:
            self.model.fit(X)
        if self.goal == 'regression':
            y_pred = self.fitted_model.predict(y)
        elif self.goal == 'classification':
            y_pred = self.fitted_model.predict_proba(y)
        result = eval_func(y, y_pred, **eval_params)
        return result
    
    def main(self, X_train, y_train, X_test, y_test, eval_func):
        
        train_result = self.sklearn_result(eval_func, None, y_train, refit=False)
        test_result = self.sklearn_result(eval_func, X_test, y_test, refit=True)
        
        return
    
    