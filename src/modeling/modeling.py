import logging
import pickle

from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from config.config import ROOT_DIR, VAL_PARAMS


logger = logging.getLogger(__name__)


class Modeling:
    """Parent modeling class."""

    def __init__(self, estimator, model_name: str='xgboost', goal: str='regression',
                 load_filepath=None, load_version=None):
        self.estimator = estimator
        self.model_name = model_name
        self.goal = goal
        if load_filepath or load_version:
            self.model = self.load_model(version=load_version, filepath=load_filepath)
        else:
            self.model = self.estimator()

    def prepare_data(self, data, features, response):

        split_params = VAL_PARAMS['split_params']
        if self.goal == 'classification':
            split_params['stratify'] = data[response]
        X_train, y_train, X_test, y_test = train_test_split(
            data[features],
            data[response],
            shuffle=True,
            **split_params
        )
        
        return X_train, y_train, X_test, y_test
    
    def save_model(self, version=0, filepath=None):
        """Save unfitted estimator with tuned hyperparameters."""
        check_is_fitted(self.model)
        if not filepath:
            filepath = ROOT_DIR / f'model/{self.model_name}_v{version}.pkl'
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
            logger.info("Model saved.")

    def load_model(self, version=0, filepath=None):
        """Load saved estimator."""
        if not filepath:
            filepath = ROOT_DIR / f'model/{self.model_name}_v{version}.pkl'
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded.")
        return model


if __name__ == '__main__':

    pass
