from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    """
    A class for training and evaluating machine learning models using scikit-learn.

    Attributes:
    -----------
    model : sklearn.base.BaseEstimator
        An instance of a scikit-learn classifier.

    Methods:
    --------
    __init__(Classifier, params)
        Constructor for ModelTrainer. Initializes with a classifier and its parameters.

    train(X_train, y_train, use_grid_search=False, grid_params=None)
        Trains the model using the provided training data. Optionally performs grid search.

    evaluate(X_test, y_test)
        Evaluates the model on the test dataset and returns a dictionary with accuracy, report, and predictions.

    save_model(filename)
        Saves the trained model to a file.

    load_model(filename)
        Loads a model from a file.
    """

    def __init__(self, Classifier, params):
        """
        Initializes the ModelTrainer with a classifier and its parameters.

        Parameters:
        -----------
        Classifier : sklearn.base.BaseEstimator
            A scikit-learn classifier class.

        params : dict
            Parameters to initialize the classifier.
        """
        self.model = Classifier(**params)

    def train(self, X_train, y_train, use_grid_search=False, grid_params=None):
        """
        Trains the model using the provided training data.

        If use_grid_search is True and grid_params is provided, performs a grid search
        to find the best parameters.

        Parameters:
        -----------
        X_train : array-like
            Training data features.

        y_train : array-like
            Training data labels.

        use_grid_search : bool, optional
            Whether to use GridSearchCV for parameter tuning (default is False).

        grid_params : dict, optional
            Parameters for GridSearchCV if use_grid_search is True.
        """
        if use_grid_search and grid_params:
            grid_search = GridSearchCV(self.model, grid_params, cv=5)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
        else:
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on the test dataset.

        Parameters:
        -----------
        X_test : array-like
            Test data features.

        y_test : array-like
            Test data labels.

        Returns:
        --------
        dict
            A dictionary containing the model's accuracy, classification report, and predictions.
        """
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        return {"accuracy": accuracy, "report": report, "predictions": predictions}

    def save_model(self, filename):
        """
        Saves the trained model to a file.

        Parameters:
        -----------
        filename : str
            The path and filename where the model should be saved.
        """
        dump(self.model, filename)

    def load_model(self, filename):
        """
        Loads a model from a file.

        Parameters:
        -----------
        filename : str
            The path and filename of the model to load.

        Returns:
        --------
        sklearn.base.BaseEstimator
            The loaded model.
        """
        self.model = load(filename)
        return self.model
