"""
Train, validation, test pipeline.
Methods with names starting with 'fit' will modify the model object.
Methods with parameter 'tune' set to True will modify the best_param object.
Method 'tune_cv' will modify the best_param object.

"""

import functools
import logging
import pickle
import warnings

import numpy as np
import pandas as pd
from hyperopt import tpe, hp, fmin, space_eval, STATUS_OK, Trials
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             mean_absolute_percentage_error,
                             r2_score,
                             log_loss,
                             f1_score,
                             precision_score,
                             recall_score,
                             accuracy_score,
                             make_scorer
)
from sklearn.model_selection import (KFold,
                                     StratifiedKFold,
                                     cross_validate,
                                     cross_val_score
) 
from typing import Union, List
from config.config import PARAMS, ROOT_DIR


logger = logging.getLogger(__name__)
warnings.filterwarnings(action='ignore', category=UserWarning)


class Pipeline:
    
    def __init__(self, X_train, y_train,
                 X_test, y_test,
                 estimator, # sklearn estimator
                 model_name: str,
                 cv_method: str='kfold',
                 n_fold: int=3,
                 scoring: str="mse", # compatible with cross_val_score
                 cv_scoring: Union[str, List]="mse", # compatible with cross_validate
                 seed=0 # does not include random_state in estimator
                 ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.estimator = estimator
        self.model = None
        self.model_name = model_name
        self.method = cv_method
        self.n_fold = n_fold
        self.scoring = scoring
        self.cv_scoring = cv_scoring
        self.seed = seed
        self.best_param = None
        self._initialize_params(self.model_name)
    
    def _initialize_params(self, model_name):
        tuning_params = PARAMS.get(model_name, {})
        for key in tuning_params:
            if type(tuning_params[key]) != list:
                pass
            elif tuning_params[key][0] == "uniform":
                tuning_params[key] = hp.uniform(key, *tuning_params[key][1])
            elif tuning_params[key][0] == "integer":
                tuning_params[key] = hp.choice(key, np.arange(*tuning_params[key][1], dtype=int))
            elif tuning_params[key][0] == "discrete":
                tuning_params[key] = hp.choice(key, np.arange(*tuning_params[key][1], dtype=float))
            elif tuning_params[key][0] == "quniform":
                tuning_params[key] = hp.quniform(key, *tuning_params[key][1])
        self.tuning_params = tuning_params
        
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
        
        # time series
        elif method == 'custom':
            cv = "custom"
        
        return cv
    
    def _make_metric(self, scorer, func, **kwargs):
        if scorer:
            return make_scorer(func, **kwargs)
        else:
            kwargs.pop('greater_is_better', None)
            return functools.partial(func, **kwargs)
    
    def _metric(self, scorer=True, score='mse'):
        # returns callable, list/dict of callables
        
        all_metrics = [
            'mse', 'r2', 'mae', 'mape', 'rmse',
            'f1_binary', 'f1_micro', 'f1_macro',
            'precision_binary', 'precision_micro', 'precision_macro',
            'recall_binary', 'recall_micro', 'recall_macro',
            'accuracy', 'log_loss'
        ]
        
        if type(score) == str:
            score = [score]
        
        assert len(set(score)) == len(set(score).intersection(set(all_metrics)))
        
        dic = dict()
        for s in score:
            if s == 'r2':
                dic[s] = self._make_metric(scorer=scorer, func=r2_score)
            elif s == 'mse':
                dic[s] = self._make_metric(scorer=scorer, func=mean_squared_error,
                                           greater_is_better=False)
            elif s == 'mae':
                dic[s] = self._make_metric(scorer=scorer, func=mean_absolute_error,
                                           greater_is_better=False)
            elif s == 'rmse':
                dic[s] = self._make_metric(scorer=scorer, func=mean_squared_error,
                                           greater_is_better=False, squared=False)
            elif s == 'mape':
                dic[s] = self._make_metric(scorer=scorer, func=mean_absolute_percentage_error,
                                           greater_is_better=False)
            elif s.startswith('f1'):
                dic[s] = self._make_metric(scorer=scorer, func=f1_score, 
                                           average=s[3:])
            elif s.startswith('precision'):
                dic[s] = self._make_metric(scorer=scorer, func=precision_score,
                                           average=s[len('precision') + 1:])
            elif s.startswith('recall'):
                dic[s] = self._make_metric(scorer=scorer, func=recall_score,
                                           average=s[len('recall') + 1:])
            elif s == 'accuracy':
                dic[s] = self._make_metric(scorer=scorer, func=accuracy_score)
            elif s == 'log_loss':
                dic[s] = self._make_metric(scorer=scorer, func=log_loss, 
                                           greater_is_better=False)

        if len(dic) == 1:
            return list(dic.values())[0]
        else:
            return dic      
    
    def _hyperparameter_tuning(self, params):
        cv = self._split(self.method, self.n_fold)
        clf = self.estimator(**params)
        scoring_scheme = self._metric(scorer=True, score=self.scoring)
        
        assert type(scoring_scheme) != dict
        
        acc = cross_val_score(clf, self.X_train, self.y_train, # TODO
                              scoring=scoring_scheme,
                              cv=cv,
                              n_jobs=-1
                             ).mean()
        
        return {"loss": -acc, "status": STATUS_OK}
    
    def _create_new_model(self, load=False, tune=False, cv_iter=20,
                     filepath=None, best_params={}):
        if load:
            model = self.load_model(filepath)
        else:
            if tune:
                self.tune_cv(n_iter=cv_iter)
                best_params = self.best_param
            else:
                if not best_params:
                    if self.best_param is not None:
                        best_params = self.best_param
                    else:
                        logger.warning("Params are empty.")
        model = self.estimator(**best_params)
        return model
    
    def tune_cv(self, n_iter=20):
        """Hyperparameter tuning. Does NOT update model.
        Updates best parameters.
        """
        trials = Trials()
        best = fmin(
            fn=self._hyperparameter_tuning,
            space=self.tuning_params,
            algo=tpe.suggest, 
            max_evals=n_iter, 
            trials=trials,
            rstate=np.random.RandomState(self.seed)
        )
        
        best_parameters = space_eval(self.tuning_params, best)
        self.best_param = best_parameters
        logger.info("Best hyperparameters updated.")
        
        best_result = trials.best_trial['result']
        logger.info(f"Best loss: {best_result['loss']}")
        return trials.results, best_result['loss']
    
    def fit(self, X, y, load=False, tune=False, cv_iter=20,
            filepath=None, best_params={}):
        """Fit model. If hyperparameters not specified,
        have the option to tune the model first.
        """
        model = self._create_new_model(load=load,
                          tune=tune,
                          cv_iter=cv_iter,
                          filepath=filepath,
                          best_params=best_params
                          )
        self.model = model
        logger.info("Model reset - ready for fitting.")
        self.model.fit(X, y)
        logger.info("Model fitted.")
        
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
    
    def fit_predict(self, X, y, X_for_pred, load=False, cv_iter=20,
                    prob=False, tune=False, filepath=None, best_params={}):
        """Retrain model on X and y with specified hyperparameters,
        then predict on X_for_pred.
        """
        
        self.fit(X, y, load=load, tune=tune, cv_iter=cv_iter,
                 filepath=filepath, best_params=best_params)
        if prob:
            y_pred = self.model.predict_proba(X_for_pred)
        else:
            y_pred = self.model.predict(X_for_pred)
        return y_pred
    
    def fit_test(self, scores, tune=False, load=False, cv_iter=20,
                 filepath=None, best_params={}):
        """Retrain model on whole train set with specified hyperparameters,
        then test on hold-out test set. 
        """
        if "log_loss" in scores:
            prob = True
        y_pred = self.fit_predict(self.X_train, self.y_train,
                                  self.X_test, load=load, 
                                  cv_iter=cv_iter, prob=prob,
                                  tune=tune, filepath=filepath,
                                  best_params=best_params
                                  )
        scoring = self._metric(scorer=False, score=scores)
        if type(scoring) == dict:
            result = {k: v(self.y_test, y_pred) for k, v in scoring.items()}
        else:
            result = {scores: scoring(self.y_test, y_pred)}

        return result, y_pred
        
    def save_model(self, version=0, filepath=None):
        """Save unfitted estimator with tuned hyperparameters."""
        if self.best_param is None:
            logger.error("No hyperparameters available for saving.")
        else:
            model = self.estimator(**self.best_param)
            if not filepath:
                filepath = ROOT_DIR / f'model/{self.model_name}_v{version}.pkl'
            with open(filepath, "wb") as f:
                pickle.dump(model, f)
            
    def load_model(self, filepath):
        """Load unfitted estimator with tuned hyperparameters."""
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded.")
        return model
    

if __name__ == '__main__':
    logger.debug("check logger printed to file.")
    logger.info("check logger printed to file and console.")