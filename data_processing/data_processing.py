from pandas import DataFrame, concat
from xml.etree.ElementTree import ParseError, fromstring
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def extract_reviews_and_ratings_to_dataframe(file_path) -> DataFrame:
    """
    Extracts reviews and ratings from an XML file and returns them as a pandas DataFrame.

    Parameters:
    file_path (str): The path to the XML file.

    Returns:
    pandas.DataFrame: A DataFrame containing the extracted reviews and ratings.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            xml_content = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            xml_content = file.read()

    reviews = xml_content.split('</review>')[:-1]
    data = []

    for review in reviews:
        try:
            review_xml = fromstring(review + '</review>')
            review_text = review_xml.find('review_text').text.strip() if review_xml.find('review_text') is not None else ''
            rating = review_xml.find('rating').text.strip() if review_xml.find('rating') is not None else ''
            data.append({'review_text': review_text, 'rating': rating})
        except ParseError:
            continue

    return DataFrame(data)

def balance_dataset(df, threshold=100000) -> DataFrame:
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

    return concat(resampled_dfs)

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