from typing import Optional
from pandas import DataFrame, concat
from sklearn.utils import resample
from numpy import percentile


def balance_dataframe(df: DataFrame, class_column: str, threshold: Optional[int] = None, random_state: Optional[int] = 123) -> DataFrame:
    """
    Balances the given DataFrame by resampling the minority classes to match the size of the majority class.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        class_column (str): The name of the column containing the class labels.
        threshold (int, optional): The minimum size of the minority class. If not provided, it is calculated as the first quartile size of the class sizes.
        random_state (int, optional): The random seed for reproducibility.

    Returns:
        pandas.DataFrame: The balanced DataFrame.

    """
    # Calculate first quartile size if threshold is not provided
    if threshold is None:
        class_sizes = df[class_column].value_counts()
        threshold = percentile(class_sizes, 5)

    # Split the DataFrame into a dictionary of DataFrames for each class
    class_dfs = {cls: df[df[class_column] == cls] for cls in df[class_column].unique()}

    # Resample each DataFrame
    resampled_dfs = []
    for cls, cls_df in class_dfs.items():
        resampled_cls_df = resample(cls_df,
                                    replace=len(cls_df) < threshold,
                                    n_samples=int(threshold),
                                    random_state=random_state)
        resampled_dfs.append(resampled_cls_df)

    # Combine into a new DataFrame and reset index
    balanced_df = concat(resampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return balanced_df