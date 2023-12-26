import xml.etree.ElementTree as ET
import pandas as pd

def extract_reviews_and_ratings_to_dataframe(file_path) -> pd.DataFrame:
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
            review_xml = ET.fromstring(review + '</review>')
            review_text = review_xml.find('review_text').text.strip() if review_xml.find('review_text') is not None else ''
            rating = review_xml.find('rating').text.strip() if review_xml.find('rating') is not None else ''
            data.append({'review_text': review_text, 'rating': rating})
        except ET.ParseError:
            continue

    return pd.DataFrame(data)