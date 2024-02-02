"""
This module contains utility functions for training, evaluating, saving,
and loading machine learning models.
"""
from sentiment_analysis.models.model_trainer import ModelTrainer

def train_model(x_train, y_train, model_class, model_params=None, param_grid=None):
    """
    Trains a machine learning model using the given training data.

    Parameters:
    x_train (array-like): The input features for training.
    y_train (array-like): The target labels for training.
    model_class (class): The class of the machine learning model to be trained.
    model_params (dict, optional): Additional parameters to be passed to the model constructor. 
    param_grid (dict, optional): Grid of hyperparameters to be searched over during training. 

    Returns:
    trainer (ModelTrainer): The trained model trainer object.
    """
    if model_params is None:
        model_params = {}
    
    trainer = ModelTrainer(model_class, model_params)
    if param_grid:
        trainer.train(x_train, y_train, use_grid_search=True, grid_params=param_grid)
    else:
        trainer.train(x_train, y_train)
    return trainer

def evaluate_model(trainer, x_test, y_test):
    """
    Evaluates the trained model on the test data.

    Args:
        trainer: The trained model.
        x_test: The input test data.
        y_test: The target test data.

    Returns:
        The evaluation results.
    """
    return trainer.evaluate(x_test, y_test)

def save_model(trainer, filename):
    """
    Save the trained model to a file.

    Args:
        trainer (Trainer): The trainer object containing the trained model.
        filename (str): The name of the file to save the model to.

    Returns:
        None
    """
    trainer.save_model(filename)

def load_model(trainer, filename):
    """
    Load a trained model from a file.

    Args:
        trainer (Trainer): The trainer object used to train the model.
        filename (str): The name of the file containing the trained model.

    Returns:
        Model: The loaded trained model.
    """
    return trainer.load_model(filename)

def train_evaluate_save_load_model(
    x_train,
    y_train,
    x_test,
    y_test,
    model_class,
    model_params=None,
    param_grid=None,
    model_name=''
    ):
    """
    Trains, evaluates, saves, and loads a machine learning model.

    Parameters:
    - x_train (array-like): Training data features.
    - y_train (array-like): Training data labels.
    - x_test (array-like): Testing data features.
    - y_test (array-like): Testing data labels.
    - model_class (class): Class of the machine learning model.
    - model_params (dict, optional): Parameters for the model. Defaults to None.
    - param_grid (dict, optional): Grid of hyperparameters for grid search. Defaults to None.
    - model_name (str, optional): Name of the model. Defaults to ''.

    Returns:
    - trainer (ModelTrainer): Trained model trainer object.
    - metrics (dict): Evaluation metrics of the model.
    - model (object): Loaded model object.
    """
    if model_params is None:
        model_params = {}
        
    trainer = ModelTrainer(model_class, model_params)

    if param_grid:
        trainer.train(x_train, y_train, use_grid_search=True, grid_params=param_grid)
    else:
        trainer.train(x_train, y_train)

    metrics = trainer.evaluate(x_test, y_test)

    model_filename = f'{model_name}_model.pkl'
    trainer.save_model(model_filename)

    model = trainer.load_model(model_filename)

    return trainer, metrics, model
