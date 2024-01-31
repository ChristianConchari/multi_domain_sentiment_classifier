
from pandas import DataFrame, Series
from sklearn.metrics import classification_report, accuracy_score
from numpy import ndarray

def add_model_report(y_true: Series, y_pred: ndarray, model_name: str) -> DataFrame:
    """
    Generates a model report DataFrame based on the true labels, predicted labels, and model name.

    Args:
        y_true (pandas.Series): The true labels.
        y_pred (numpy.ndarray): The predicted labels.
        model_name (str): The name of the model.

    Returns:
        pandas.DataFrame: The model report DataFrame.
    """
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    model_report_df = DataFrame(report_dict).transpose()

    accuracy = accuracy_score(y_true, y_pred)

    model_report_df['accuracy'] = accuracy

    model_name_formatted = f"{model_name}"

    model_report_df['model_name'] = model_name_formatted

    model_report_df.reset_index(inplace=True)

    model_report_df.rename(columns={'index': 'class_label'}, inplace=True)

    columns_order = ['model_name', 'class_label', 'accuracy'] + [col for col in model_report_df if col not in ['model_name', 'class_label', 'accuracy']]

    model_report_df = model_report_df[columns_order]

    model_report_df = model_report_df[~model_report_df['class_label'].str.contains("avg")]

    model_report_df = model_report_df[model_report_df['class_label'].apply(lambda x: x.isnumeric())]

    return model_report_df

