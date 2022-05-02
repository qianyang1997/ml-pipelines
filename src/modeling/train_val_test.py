"""
Train, validation, test pipeline.
Methods with names starting with 'fit' will modify the model object.
Methods with names starting with 'tune' or have parameter 'tune' set
to True will modify the best_param object.
"""
from functools import partial
import logging
import pickle
from typing import Union, List

import numpy as np
import pandas as pd
from hyperopt import tpe, hp, fmin, space_eval, STATUS_OK, Trials
from sklearn.model_selection import (KFold,
                                     StratifiedKFold,
                                     cross_validate,
                                     cross_val_score,
                                     )
from sklearn.utils.validation import check_is_fitted
from config.config import MODEL_PARAMS, VAL_PARAMS, ROOT_DIR


logger = logging.getLogger(__name__)


class Pipeline:
    """Pipeline to perform train-test-validation with specified estimator."""
    def __init__(self, estimator, model_name: str='xgboost'):
        self.estimator = estimator
        self.model = self.estimator()
        self.model_name = model_name

    def _initialize_params(self):
        tuning_params = MODEL_PARAMS.get(self.model_name, {})
        for key in tuning_params:
            if not isinstance(tuning_params[key], list):
                pass
            elif tuning_params[key][0] == "uniform":
                tuning_params[key] = hp.uniform(key, *tuning_params[key][1])
            elif tuning_params[key][0] == "integer":
                tuning_params[key] = hp.choice(key, np.arange(*tuning_params[key][1], dtype=int))
            elif tuning_params[key][0] == "discrete":
                tuning_params[key] = hp.choice(key, np.arange(*tuning_params[key][1], dtype=float))
            elif tuning_params[key][0] == "quniform":
                tuning_params[key] = hp.quniform(key, *tuning_params[key][1])
        return tuning_params

    # TODO - make more flexible
    def _split(self, method='kfold', n_fold=3):
        # k fold (specify n folds)
        if method == 'kfold':
            cv = KFold(n_splits=n_fold,
                    shuffle=True,
                    random_state=self.seed)
        # stratified k fold
        elif method == 'kfold_classification':
            cv = StratifiedKFold(n_splits=n_fold,
                                shuffle=True,
                                random_state=self.seed)
        # time series - TODO
        elif method == 'custom':
            cv = "custom"
        return cv

    def _hyperparameter_tuning(self, X_train, y_train, score_scheme, params):
        cv = self._split(self.method, self.n_fold) # TODO
        clf = self.estimator(**params)

        acc = cross_val_score(clf,
                              X_train,
                              y_train,
                              scoring=score_scheme,
                              cv=cv,
                              n_jobs=-1
                             ).mean()

        return {"loss": -acc, "status": STATUS_OK}

    def tune_cv(self, X_train, y_train):
        """Hyperparameter tuning. Returns best result and tuning params."""
        
        cv_params = VAL_PARAMS['cv_params']
        
        trials = Trials()
        
        func = partial(self._hyperparameter_tuning,
                       X_train, y_train,
                       cv_params['evaluation'])
        tuning_params = self._initialize_params(self.model_name)
        
        best = fmin(
            fn=func,
            space=tuning_params,
            algo=tpe.suggest,
            max_evals=cv_params['n_iter'],
            trials=trials,
            rstate=np.random.default_rng(cv_params['random_state'])
        )

        best_parameters = space_eval(tuning_params, best)
        best_result = trials.best_trial['result']
        logger.info(f"Best loss: {best_result['loss']}")
        
        return trials.results, best_result['loss'], best_parameters

    def cv_eval(self, load=False, tune=False, cv_iter=20,
                filepath=None, best_params={}):
        """Generate CV evaluation table given specified hyperparameters.
        CV folds are the same as those used in hyperparameter tuning.
        Method does not fit model instance.
        """
        model = self._create_new_model(load=load,
                          tune=tune,
                          cv_iter=cv_iter,
                          filepath=filepath,
                          best_params=best_params
                          )
        cv = self._split(self.method, self.n_fold)
        scoring_scheme = self._metric(scorer=True, score=self.cv_scoring)
        cv_score = cross_validate(model, self.X_train, self.y_train,
                                  scoring=scoring_scheme,
                                  cv=cv,
                                  n_jobs=-1,
                                  return_train_score=True
                                  )
        cv_score = pd.DataFrame(cv_score)
        return cv_score
    
    def main(self, data,
             load_version=None, load_filepath=None,
             save_version=None, save_filepath=None):
        
        # TODO - train test split
        X_train, y_train, X_test, y_test = data
        
        if load_filepath or load_version:
            self.model = self.load_model(version=load_version, filepath=load_filepath)
            check_is_fitted(self.model)
        else:
            _, best_result, best_params = self.tune_cv(X_train, y_train)
            self.model = self.estimator(**best_params).fit(X_train)

        if save_filepath or save_version:
            self.save_model(version=save_version, filepath=save_filepath)
        
        # TODO - evaluate performance on train set and test set
        

    def save_model(self, version=0, filepath=None):
        """Save unfitted estimator with tuned hyperparameters."""
        check_is_fitted(self.model)
        if not filepath:
            filepath = ROOT_DIR / f'model/{self.model_name}_v{version}.pkl'
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, version=0, filepath=None):
        """Load saved estimator."""
        if not filepath:
            filepath = ROOT_DIR / f'model/{self.model_name}_v{version}.pkl'
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Model loaded.")


if __name__ == '__main__':

    logger.debug("check logger printed to file.")
    logger.info("check logger printed to file and console.")
