import re
import logging
import pandas as pd
import numpy as np

import sklearn.linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from src.custom_folds import time_series_split, time_folds

logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

eval_metrics = {'mse': make_scorer(mean_squared_error, greater_is_better=False),
                'r2': make_scorer(r2_score)}


class Modeling:
    """
    Modeling pipeline with time series cv.
    """

    def __init__(self, dat: pd.DataFrame, fea: list, pk: list, res: str) -> None:
        """
        Load data based on selected features.
        :param dat: pandas dataframe - must have a 'ym' column;
               indicate year and month of each record.
        :param fea: list - features used for modeling.
        :param pk: list - primary keys of the dataset - must include 'ym'.
        :param res: str - response variable for modeling.
        """
        self.data = dat.copy()  # so original will not be changed
        self.data.ym = pd.to_datetime(self.data.ym).dt.to_period('M')
        self.keys = pk
        self.response = [res]
        self.features = fea

        if 'ym' not in self.keys:
            raise ValueError('Primary key must include "ym".')

    def filter_data(self, start_ym: str, end_ym: str) -> pd.DataFrame:
        """
        Subset data based on dates.
        :param start_ym: str - start date, in '%Y-%m' format.
        :param end_ym: str - end date, in '%Y-%m' format.
        :return: pandas dataframe - subset data.
        """
        return self.data[(self.data.ym >= start_ym) & (self.data.ym <= end_ym)]

    def cv(self, cv_method: str, func_name: str, folds: dict, csv: pd.DataFrame,
           param_distributions: dict, random_search: bool,
           params: dict, func: object, refit: str = 'mse', **kwargs) -> tuple:
        """
        Perform time series cross validation.
        :param cv_method: str - can be 'expanding_window', 'sliding_window', or 'step_by_step'.
        :param func_name: str - name of the function (custom defined).
        :param folds: dict - dictionary of indexes in custom-defined folds.
        :param csv: pandas DataFrame - data to perform cv on.
        :param param_distributions: dict - hyperparameters to tune - see sklearn's randomSearchCV
               documentation. Cannot be null if random_search == True.
        :param random_search: bool - if True, uses sklearn's randomSearchCV functionality.
        :param params: dict - hyperparameters to use - see sklearn's cross_validate documentation.
               Only used when random_search == False.
        :param func: obj - sklearn or sklearn-compatible model.
        :param refit: str - error metric. Can be 'mse' or 'r2'.
        :param kwargs: can specify n_iter, random_state, etc. See RandomizedSearchCV documentation.
        :return: tuple (sklearn estimator, pandas dataframe) - a fitted model and best
                 chosen hyperparameters organized in a dataframe.
        """
        if refit not in ['mse', 'r2']:
            raise ValueError('Error metric must be mse or r2. See documentation.')

        if cv_method not in ['sliding_window', 'expanding_window', 'step_by_step']:
            raise ValueError('Invalid CV method. See documentation.')

        def expanding_window():
            for i in range(len(folds) - 1):
                train_csv = csv[(csv['ym'] >= folds[0][0]) & (csv['ym'] <= folds[i][-1])]. \
                    reset_index(drop=True)
                valid_csv = csv[(csv['ym'] >= folds[i + 1][0]) & (csv['ym'] <= folds[i + 1][-1])]. \
                    reset_index(drop=True)
                train_index = train_csv.index.tolist()
                val_index = valid_csv.index.tolist()
                logger.debug(f'Time periods in split{i}:'
                             f' train - {folds[0][0]}~{folds[i][-1]};'
                             f' val - {folds[i + 1][0]}~{folds[i + 1][-1]}\n'
                             f'Number of data points in split{i}:'
                             f' train - {len(train_index)};'
                             f' val - {len(val_index)}')
                yield train_index, val_index

        def sliding_window():
            for i in range(len(folds) - 1):
                train_csv = csv[(csv['ym'] >= folds[i][0]) & (csv['ym'] <= folds[i][-1])]. \
                    reset_index(drop=True)
                valid_csv = csv[(csv['ym'] >= folds[i + 1][0]) & (csv['ym'] <= folds[i + 1][-1])]. \
                    reset_index(drop=True)
                train_index = train_csv.index.tolist()
                val_index = valid_csv.index.tolist()
                logger.debug(f'Time periods in split{i}:'
                             f' train - {folds[i][0]}~{folds[i][-1]};'
                             f' val - {folds[i + 1][0]}~{folds[i + 1][-1]}\n'
                             f'Number of data points in split{i}:'
                             f' train - {len(train_index)};'
                             f' val - {len(val_index)}')
                yield train_index, val_index

        def step_by_step():
            for i in range(len(folds)):
                train_csv = csv[(csv['ym'] >= folds[i][0]) & (csv['ym'] <= folds[i][1])]. \
                    reset_index(drop=True)
                valid_csv = csv[(csv['ym'] >= folds[i][2]) & (csv['ym'] <= folds[i][3])]. \
                    reset_index(drop=True)
                train_index = train_csv.index.tolist()
                val_index = valid_csv.index.tolist()
                logger.debug(f'Time periods in split{i}:'
                             f' train - {folds[i][0]}~{folds[i][1]};'
                             f' val - {folds[i][2]}~{folds[i][3]}\n'
                             f'Number of data points in split{i}:'
                             f' train - {len(train_index)};'
                             f' val - {len(val_index)}')
                yield train_index, val_index

        if cv_method == 'expanding_window':
            cv = expanding_window()
        elif cv_method == 'sliding_window':
            cv = sliding_window()
        else:
            cv = step_by_step()

        if random_search:
            assert param_distributions is not None
            func_instance = func()
            fit = RandomizedSearchCV(func_instance,
                                     param_distributions=param_distributions,
                                     refit=refit,
                                     scoring=eval_metrics,
                                     cv=cv,
                                     **kwargs)
            fit.fit(csv[self.features], csv[self.response].values.reshape(-1, 1))
            estimator = fit.best_estimator_

            # output a dataframe of scores of each fold and mean aggregated scores;
            # output the best refitted score using optimal parameters
            results = pd.DataFrame(fit.cv_results_). \
                filter(regex='^((split)|(mean_test)|(std_test)|(params))') \
                .rename(columns=lambda x: re.sub('test', 'val', x))

            if refit == 'mse':
                result1 = results.iloc[np.where(results['mean_val_mse'] == fit.best_score_)]
            else:
                result1 = results.iloc[np.where(results['mean_val_r2'] == fit.best_score_)]

            logger.debug(f'CV result for {func_name}: \n{results}\n')
            logger.info(f'\nBest parameters from cv are: {fit.best_params_}\n'
                        f'Best scores from cv are: {fit.best_score_}')

        else:
            if params is None:
                params = {}

            estimator = func(**params)
            cv_results = cross_validate(estimator,
                                        scoring=eval_metrics,
                                        X=csv[self.features],
                                        y=csv[self.response].values.reshape(-1, 1),
                                        cv=cv,
                                        **kwargs)
            # refit the model with hyperparameters on the whole train set
            estimator.fit(csv[self.features], csv[self.response].values.reshape(-1, 1))

            # if a linear model is fitted, log the coefficients
            if isinstance(estimator, (sklearn.linear_model._base.LinearRegression,
                                sklearn.linear_model._coordinate_descent.Lasso,
                                sklearn.linear_model._ridge.Ridge)):
                coefficients = estimator.coef_
            else:
                coefficients = None

            # calculate mean, std test(val) mse and r2
            results = pd.DataFrame(cv_results)[['test_mse', 'test_r2']]
            mean_val_mse = results['test_mse'].mean()
            std_val_mse = results['test_mse'].std()
            mean_val_r2 = results['test_r2'].mean()
            std_val_r2 = results['test_r2'].std()

            # flatten the matrix, column='split{i}_val_mse' or 'split{i}_val_r2'
            number_of_folds = results.shape[0]
            result_flattened = results.T.values.reshape(1, -1)
            col = [f'split{i}_val_mse' for i in range(number_of_folds)] + \
                  [f'split{i}_val_r2' for i in range(number_of_folds)]

            result1 = pd.DataFrame(result_flattened, columns=col)
            result1['mean_val_mse'] = mean_val_mse
            result1['mean_val_r2'] = mean_val_r2
            result1['std_val_mse'] = std_val_mse
            result1['std_val_r2'] = std_val_r2
            result1['params'] = f'{params}'

            logger.debug(f'CV result for {func_name}: \n{pd.DataFrame(results)}\n')

            if coefficients is not None:
                logger.info(f'Coefficients are:\n'
                            f'{pd.Series(coefficients[0], index=self.features)}\n')

        # return the estimator fitted on the entire train set with specified parameters
        return estimator, result1

    def test_model(self, func_name: str, func: object, test_set: pd.DataFrame) -> tuple:
        """
        Predict on test set.
        :param func_name: str - name of the fitted model.
        :param func: obj - a fitted sklearn/sklearn-compatible model.
        :param test_set: pandas dataframe - test set.
        :return: tuple (float, float) - error metrics.
        """
        y_pred = func.predict(test_set[self.features])
        mse = mean_squared_error(test_set[self.response], y_pred)
        r2 = r2_score(test_set[self.response], y_pred)
        logger.debug(f'Test result for {func_name}: MSE = {mse}, R2 = {r2}\n')
        return mse, r2

    def predict_future(self, func: object, train_start: str, train_end: str,
                       new_ym_start: str, new_ym_end: str, **kwargs) -> np.array:
        """
        Predict on unlabeled rows.
        :param func: obj - an unfitted sklearn/sklearn-compatible model.
        :param train_start: str - train set start date, in '%Y-%m' format.
        :param train_end: str - train set end date, in '%Y-%m' format.
        :param new_ym_start: new rows start date, in '%Y-%m' format.
        :param new_ym_end: new rows end date, in '%Y-%m' format.
        :param kwargs: hyperparameters for the sklearn model.
        :return: numpy array - array of predicted values.
        """
        train_data = self.filter_data(train_start, train_end)[self.features + self.response]
        new_data = self.filter_data(new_ym_start, new_ym_end)[self.features]
        final_model = func(**kwargs)
        final_model.fit(train_data[self.features], train_data[self.response].values.reshape(-1, 1))
        y_pred = final_model.predict(new_data[self.features])
        return y_pred

    def main(self, n_fold: int, func: object, cv_method: str, func_name: str,
             train_start: str, train_end: str, test_start: str, test_end: str,
             params: dict = None, param_distributions: dict = None,
             refit: str = 'mse', random_search: bool = True, step: int = 0,
             **kwargs) -> pd.DataFrame:
        """
        Pipeline for performing cv on train set and testing the fitted model on test set.
        :param n_fold: int - number of cv folds.
        :param func: object - a sklearn/sklearn-compatible model.
        :param cv_method: str - can be 'sliding_window', 'expanding_window', or 'step_by_step'.
        :param func_name: str - name of the model (custom defined).
        :param train_start: str - train set start date, in '%Y-%m' format.
        :param train_end: str - train set end date, in '%Y-%m' format.
        :param test_start: str - test set start date, in '%Y-%m' format.
        :param test_end: str - test set end date, in '%Y-%m' format.
        :param params: dict - hyperparameters for the model. See the 'cv' method for details.
        :param param_distributions: dict - hyperparameters to tune for the model.
               See the 'cv' method for details.
        :param refit: str - error metric. Can be 'mse' or 'r2'.
        :param random_search: bool - see the 'cv' method for details.
        :param step: int - time sliding frame for performing step-by-step cv method.
               If 'step' < 1, time_series_split would be used to create cv folds.
               Otherwise, time_folds is used.
        :param kwargs: hyperparameters for either randomSearchCV or cross_validate.
               See the 'cv' method and original sklearn documentation for details.
        :return: pandas dataframe - result of best combination of hyperparameters.
        """
        # split data into train and test
        train_set = self.filter_data(train_start, train_end)[self.keys +
                                                             self.features +
                                                             self.response]
        test_set = self.filter_data(test_start, test_end)[self.features +
                                                          self.response]

        # identify custom CV folds
        if step < 1:
            folds = time_series_split(n_fold, train_start, train_end)
        else:
            folds = time_folds(n_fold=n_fold, start_ym=train_start,
                               end_ym=train_end, steps=step)

        # perform CV on train set - log CV and test results, return fitted models
        tuned_model, best_result = self.cv(cv_method=cv_method,
                                           func=func,
                                           func_name=func_name,
                                           folds=folds,
                                           csv=train_set,
                                           param_distributions=param_distributions,
                                           random_search=random_search,
                                           params=params,
                                           refit=refit,
                                           **kwargs)

        best_result['cv_method'] = cv_method
        best_result['train_period'] = f'{train_start}~{train_end}'
        best_result['test_period'] = f'{test_start}~{test_end}'
        best_result['model'] = func_name

        # test the model
        test_mse, test_r2 = self.test_model(func_name, tuned_model, test_set)

        best_result['test_mse'] = test_mse
        best_result['test_r2'] = test_r2

        return best_result


if __name__ == '__main__':
    # an example using a linear model for cross validation
    data = pd.read_csv('some_data.csv').dropna(how='any')
    features = ['feature1', 'feature2']
    keys = ['key1', 'ym']
    response = 'response'

    # initiate Modeling object
    obj = Modeling(data, features, features, keys, response)

    # run main method
    result = obj.main(n_fold=10,
                      func=LinearRegression,
                      cv_method='step-by-step',
                      func_name='baseline model',
                      train_start='2012-01',
                      train_end='2020-12',
                      test_start='2019-01',
                      test_end='2019-02',
                      random_search=False,
                      step=1)
    pd.set_option('display.max_columns', None)
    print(result)
