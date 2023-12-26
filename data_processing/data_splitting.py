from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=42) -> tuple:
    """
    Split the data into training and testing sets.

    Parameters:
    X (array-like): The input features.
    y (array-like): The target variable.
    test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
    random_state (int, optional): The seed used by the random number generator. Defaults to 42.

    Returns:
    X_train (array-like): The training features.
    X_test (array-like): The testing features.
    y_train (array-like): The training target variable.
    y_test (array-like): The testing target variable.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
