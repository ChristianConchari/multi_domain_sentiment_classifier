from pandas import DataFrame, concat
from xml.etree.ElementTree import ParseError, fromstring
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


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

def balance_dataframe(df, class_column, threshold, random_state=123):
    """
    Balance a DataFrame by resampling classes to a threshold.

    Parameters:
    df (pd.DataFrame): The DataFrame to balance.
    class_column (str): The column name of the class labels.
    threshold (int): The number of samples each class should have.
    random_state (int): The random state for reproducibility.

    Returns:
    pd.DataFrame: The balanced DataFrame.
    """
    # Split the DataFrame into a dictionary of DataFrames for each class
    class_dfs = {cls: df[df[class_column] == cls] for cls in df[class_column].unique()}

    # Resample each DataFrame
    resampled_dfs = []
    for cls, cls_df in class_dfs.items():
        if len(cls_df) > threshold:
            # Downsample classes above the threshold
            resampled_cls_df = resample(cls_df,
                                        replace=False,
                                        n_samples=threshold,
                                        random_state=random_state)
        else:
            # Upsample classes below the threshold
            resampled_cls_df = resample(cls_df,
                                        replace=True,
                                        n_samples=threshold,
                                        random_state=random_state)
        resampled_dfs.append(resampled_cls_df)

    # Combine into a new DataFrame
    balanced_df = concat(resampled_dfs)

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