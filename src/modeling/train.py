"""
Model training pipeline.
"""
from functools import partial
import logging

import numpy as np
import pandas as pd
from hyperopt import tpe, hp, fmin, space_eval, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, cross_val_score

from config.config import MODEL_PARAMS, VAL_PARAMS
from src.modeling.modeling import Modeling


logger = logging.getLogger(__name__)


class Train(Modeling):
    """Pipeline to perform train-test-validation with specified estimator."""
    def __init__(self, estimator, model_name: str='xgboost', goal: str='regression'):
        Modeling.__init__(self, estimator, model_name, goal, None, None)

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
    def _split(self):
        split_params = VAL_PARAMS['cv_params']
        fold_params = {
            'n_splits': split_params['n_folds'],
            'shuffle': True,
            'random_state': split_params['kfold_random_state']
        }
        
        if self.goal == 'regression':
            cv = KFold(**fold_params)
        elif self.goal == 'classification':
            cv = StratifiedKFold(**fold_params)
        # time series - TODO
        
        return cv

    def _hyperparameter_tuning(self, X_train, y_train, score_scheme, params):
        cv = self._split()
        clf = self.estimator(**params)

        acc = cross_val_score(clf,
                              X_train,
                              y_train,
                              scoring=score_scheme,
                              cv=cv,
                              n_jobs=-1).mean()

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
            early_stop_fn=no_progress_loss,
            rstate=np.random.default_rng(cv_params['tuning_random_state'])
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
        cv = self._split()
        scoring_scheme = self._metric(scorer=True, score=self.cv_scoring)
        cv_score = cross_validate(model, self.X_train, self.y_train,
                                  scoring=scoring_scheme,
                                  cv=cv,
                                  n_jobs=-1,
                                  return_train_score=True
                                  )
        cv_score = pd.DataFrame(cv_score)
        return cv_score
        
    def main(self, X_train, y_train,
             save_version=None, save_filepath=None):
        
        _, best_result, best_params = self.tune_cv(X_train, y_train)
        self.model = self.estimator(**best_params).fit(X_train)

        if save_filepath or save_version:
            self.save_model(version=save_version, filepath=save_filepath)
        
        return best_result, best_params


if __name__ == '__main__':

    logger.debug("check logger printed to file.")
    logger.info("check logger printed to file and console.")
