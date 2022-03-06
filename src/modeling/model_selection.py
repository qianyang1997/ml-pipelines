"""Select models.
"""

import logging
from typing import Union, List
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import config.config
from src.modeling.train_val_test import Pipeline


logger = logging.getLogger(__name__)


class ModelSelection:
    """Select models."""

    def __init__(self,
                 data: pd.DataFrame,
                 features: Union[str, List[str], np.ndarray],
                 response: str,
                 goal: str='regression',
                 ):
        self.goal = goal
        self.data = data
        self.features = features
        self.response = response
        (self.X_train, self.y_train,
         self.X_test, self.y_test) = self.prepare_data()

    def prepare_data(self):

        X = None
        y = None
        
        # categorical variables
        
        return X, y, None, None

    def tree(self, X, y, X_test, y_test, **kwargs):

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
        return obj
        
    def baseline(self, reg_scores: Union[List[str], str, None]):
        """Returns scoring and prediction result from
           linear model if goal is regression;
           returns percent of training majority class(es)
           in test set if goal is classification.
        """
        if self.goal == 'regression':
            model = Pipeline(self.X_train,
                             self.y_train,
                             self.X_test,
                             self.y_test,
                             estimator=LinearRegression
                            )
            result, y_pred = model.fit_test(scores=reg_scores)
        elif self.goal == 'classification':
            values = pd.Series(self.y_train).value_counts()
            values_test = pd.Series(self.y_test).value_counts()
            maj_class = np.where(values == values.sort_values().tolist()[-1])
            result = {}
            for i in maj_class[0]:
                try:
                    result[i] = values_test[i] / len(self.y_test)
                except KeyError:
                    logger.error(f'Class {i} is not in y test.')
            y_pred = None

        return result, y_pred


if __name__ == '__main__':

    iris = datasets.load_iris()
    df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    
    model = ModelSelection(
        data=df,
        features=None,
        response=None,
        goal='regression',
        )
    model.X_train = df
    model.y_train = iris.target
    model.X_test = df
    model.y_test = iris.target
    print(model.baseline(['mse', 'r2']))
