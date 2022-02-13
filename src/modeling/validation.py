from sklearn.model_selection import (KFold,
                                     StratifiedKFold,
                                     RandomizedSearchCV,
                                     cross_validate
) 
from typing import Union


class Validation:
    
    def __init__(self, 
                 estimator,
                 params,
                 nested=True,
                 seed=0
                 ):
        self.model = estimator
        self.params = params
        self.seed = seed
        self.nested = nested
        
    def split(self, method='kfold', n_fold=3):
        # k fold (specify n folds)
        if method == 'kfold':
            cv = KFold(n_splits=n_fold,
                    shuffle=True,
                    seed=self.seed)
        # stratified k fold
        elif method == 'kfold_classification':
            cv = StratifiedKFold(n_splits=n_fold,
                                shuffle=True,
                                seed=self.seed)
        
        # time series
        cv = "custom"
        
        return cv
    
    def metric(self):
        # returns str, callable, list/tuple of strings, list/tuple of callables
        pass
    
    def tuning(self,
               X,
               y,
               method: Union[str, tuple]=('kfold',),
               n_fold: Union[int, tuple]=(3,),
               **kwargs):
        
        # cast param type to tuple
        if type(method) == str:
            method = (method,)
        if type(n_fold) == int:
            n_fold = (n_fold,)
        if self.nested:
            if len(method) == 1:
                method *= 2
            if len(n_fold) == 1:
                n_fold *= 2
        
        # (inner) cv
        cv = self.split(method[0], n_fold[0])
        scoring_scheme = self.metric() # TODO
        clf = RandomizedSearchCV(estimator=self.model,
                                 param_grid=self.params,
                                 cv=cv,
                                 scoring=scoring_scheme,
                                 refit=True,
                                 random_state=self.seed,
                                 **kwargs)
        
        # outer cv
        if self.nested:
            outer_cv = self.split(method[1], n_fold[1])
            outer_scoring = self.metric() # TODO
            val_score = cross_validate(clf,
                                       X=X,
                                       y=y,
                                       cv=outer_cv,
                                       scoring=outer_scoring
                                       )
        else:
            clf.fit(X=X, y=y)
            
        return clf # tuned estimator
         
        
        
        

