from pandas import DataFrame, concat
from xml.etree.ElementTree import ParseError, fromstring
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from numpy import percentile
from typing import Optional


def extract_reviews_and_ratings_to_dataframe(file_path: str, category: str) -> DataFrame:
    """
    Extracts reviews and ratings from an XML file and returns them as a DataFrame.

    Parameters:
    - file_path (str): The path to the XML file.
    - category (str): The category of the reviews.

    Returns:
    - DataFrame: A pandas DataFrame containing the extracted data, with columns for review text, rating, category, and review class.
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
            review_class = '1' if float(rating) > 3 else '0'
            data.append({'review_text': review_text, 'rating': rating, 'category': category, 'review_class': review_class})
        except ParseError:
            continue

    return DataFrame(data)

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


def split_data(df, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Parameters:
    - df: DataFrame containing the data.
    - test_size: The proportion of the data to be used for testing (default: 0.2).
    - random_state: The seed used by the random number generator (default: 42).

    Returns:
    - X_train: The training data.
    - X_test: The testing data.
    - y_train: The labels for the training data.
    - y_test: The labels for the testing data.
    """
    X = df['review_text'].tolist()
    y = df['rating']

    return train_test_split(X, y, test_size=test_size, random_state=random_state)