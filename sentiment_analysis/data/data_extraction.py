from xml.etree.ElementTree import ParseError, fromstring
from pandas import DataFrame

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