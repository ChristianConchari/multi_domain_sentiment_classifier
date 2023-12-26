import pandas as pd
from sklearn.utils import resample

def balance_dataset(df, threshold=100000) -> pd.DataFrame:
    """
    Balances the dataset by resampling the minority classes to match the size of the majority class.
    Performs both downsampling and upsampling.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing the dataset.
    threshold (int): The desired size for each class after resampling. Defaults to 100000.

    Returns:
    pandas.DataFrame: The balanced dataset.
    """
    class_dfs = {cls: df[df['rating'] == cls] for cls in df['rating'].unique()}
    resampled_dfs = []

    for cls, cls_df in class_dfs.items():
        if len(cls_df) > threshold:
            resampled_cls_df = resample(cls_df, replace=False, n_samples=threshold, random_state=123)
        else:
            resampled_cls_df = resample(cls_df, replace=True, n_samples=threshold, random_state=123)
        resampled_dfs.append(resampled_cls_df)

    return pd.concat(resampled_dfs)
