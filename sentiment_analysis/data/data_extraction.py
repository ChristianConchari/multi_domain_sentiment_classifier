from xml.etree.ElementTree import ParseError, fromstring
from pandas import DataFrame

def extract_reviews_and_ratings_to_dataframe(file_path: str, category: str) -> DataFrame:
    """
    Extracts reviews and ratings from an XML file and returns them as a DataFrame.

    Parameters:
    - file_path (str): The path to the XML file.
    - category (str): The category of the reviews.

    Returns:
    - DataFrame: A pandas DataFrame containing the extracted data, 
        with columns for review text, rating, category, and review class.
    """
    try:
        # Try to read the file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            xml_content = file.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try reading the file with ISO-8859-1 encoding
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            xml_content = file.read()

    # Split the XML content into individual reviews
    reviews = xml_content.split('</review>')[:-1]
    data = []

    for review in reviews:
        try:
            # Parse the review XML content
            review_xml = fromstring(review + '</review>')
            # Extract the review text and rating, with default values if missing
            review_text = review_xml.find('review_text').text.strip() if review_xml.find('review_text') is not None else ''
            # Extract the rating, with default value if missing
            rating = review_xml.find('rating').text.strip() if review_xml.find('rating') is not None else ''
            # Assign a review class based on the rating
            review_class = '1' if float(rating) > 3 else '0'
            # Append the extracted data to the list
            data.append({'review_text': review_text, 'rating': rating, 'category': category, 'review_class': review_class})
        except ParseError:
            continue
    return DataFrame(data)
