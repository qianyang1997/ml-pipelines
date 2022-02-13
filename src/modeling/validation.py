import logging
import pickle
import numpy as np
import pandas as pd
from hyperopt import tpe, hp, fmin, space_eval, STATUS_OK, Trials
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             mean_absolute_percentage_error,
                             r2_score,
                             make_scorer
)
from sklearn.model_selection import (KFold,
                                     StratifiedKFold,
                                     cross_validate,
                                     cross_val_score
) 
from typing import Union, List
from config.config import PARAMS, ROOT_DIR


class Validation:
    
    def __init__(self, X_train, y_train,
                 X_test, y_test,
                 estimator, # sklearn estimator
                 model_name,
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
        self.initialize_params(self.model_name)
    
    def initialize_params(self, model_name):
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
        
    def split(self, method='kfold', n_fold=3):
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
    
    def metric(self, scorer=True, score='mse'):
        # returns callable, list/dict of callables
        if type(score) == str:
            score = [score]
        if scorer:
            scorer = dict()
            for s in score:
                if s == 'r2':
                    scorer[s] = make_scorer(r2_score)
                if s == 'mse':
                    scorer[s] = make_scorer(mean_squared_error, 
                                            greater_is_better=False)
        else:
            scorer = list()
            for s in score:
                if s == 'r2':
                    scorer[s] = r2_score
                if s == 'mse':
                    scorer[s] = mean_squared_error
        if len(scorer) == 1:
            return list(scorer.values())[0]
        elif not scorer:
            assert ValueError("Must specify scoring scheme.")
        else:
            return scorer
            
    
    def _hyperparameter_tuning(self, params):
        cv = self.split(self.method, self.n_fold)
        clf = self.estimator(**params)
        scoring_scheme = self.metric(score=self.scoring) # TODO
        
        assert type(scoring_scheme) != dict
        
        acc = cross_val_score(clf, self.X_train, self.y_train, # TODO
                              scoring=scoring_scheme,
                              cv=cv,
                              n_jobs=-1
                             ).mean()
        
        return {"loss": -acc, "status": STATUS_OK}
    
    def _update_model(self, load=False, tune=False, cv_iter=20,
                     filepath=None, best_params={}):
        if load:
            self.load_model(filepath)
        else:
            if tune:
                self.fit_cv(n_iter=cv_iter)
                best_params = self.best_param
            else:
                if not best_params:
                    if self.best_param is not None:
                        best_params == self.best_param
                    else:
                        logging.warning("Params are empty.")
            self.model = self.estimator(**best_params)
            logging.info("Model updated with hyperparameters.")
    
    def fit_cv(self, n_iter=20):
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
        logging.info("Best hyperparameters updated.")
        
        best_result = trials.best_trial['result']
        return trials.results, best_result['loss']
    
    def fit(self, X, y, load=False, tune=False, cv_iter=20,
            filepath=None, best_params={}):
        """Fit model. If hyperparameters not specified,
        have the option to tune the model first.
        """
        self._update_model(load=load,
                          tune=tune,
                          cv_iter=cv_iter,
                          filepath=filepath,
                          best_params=best_params
                          )
        
        self.model.fit(X, y)
        logging.info("Model fitted.")
        
    def cv_eval(self, load=False, tune=False, cv_iter=20,
                filepath=None, best_params={}):
        """Generate CV evaluation table given specified hyperparameters.
        CV folds are the same as those used in hyperparameter tuning.
        If hyperparameters not specified, print warning message.
        """
        self._update_model(load=load,
                          tune=tune,
                          cv_iter=cv_iter,
                          filepath=filepath,
                          best_params=best_params
                          )
        cv = self.split(self.method, self.n_fold)
        scoring_scheme = self.metric(score=self.cv_scoring)
        cv_score = cross_validate(self.model, self.X_train, self.y_train,
                                  scoring=scoring_scheme,
                                  cv=cv,
                                  n_jobs=-1,
                                  return_train_score=True
                                  )
        cv_score = pd.DataFrame(cv_score)
        return cv_score
    
    def test(self, scores, load=False, filepath=None, best_params={}):
        """Retrain model on whole train set with specified hyperparameters,
        then test on holdout test set. 
        """
        
        self.fit(self.X_train, self.y_train, load=load,
                 tune=False, filepath=filepath, best_params=best_params)
        y_pred = self.model.predict(self.X_test)
        scoring = self.metric(scorer=False, score=scores)
        if type(scoring) == list:
            result = {scores[i]: scoring[i](y_true=self.y_test, y_pred=y_pred) 
                      for i in len(scores)}
        else:
            result = {scores: scoring(y_true=self.y_test, y_pred=y_pred)}

        return result
        
    def save_model(self, version=0, filepath=None):
        """Save model with hyperparameters.
        Model is unfitted once reloaded.
        """
        if self.best_param is None:
            logging.error("No hyperparameters available for saving.")
        else:
            self._update_model(load=False, tune=False, best_params={})
            if not filepath:
                filepath = ROOT_DIR / f'model/{self.model_name}_v{version}.pkl'
            with open(filepath, "wb") as f:
                pickle.dump(self.model, f)
            
    def load_model(self, filepath):
        """Load model with specified hyperparameters.
        Loaded model is unfitted.
        """
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        logging.info("Model loaded.")