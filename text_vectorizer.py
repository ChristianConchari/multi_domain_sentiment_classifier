import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load

class TextVectorizer:
    def __init__(self, use_default_stopwords=True, custom_stopwords=None):
        """
        Initialize the TextVectorizer class.

        Parameters:
        - use_default_stopwords (bool): Whether to use default English stopwords. Default is True.
        - custom_stopwords (list): List of custom stopwords. Default is None.

        Returns:
        None
        """
        if use_default_stopwords:
            nltk.download('stopwords', quiet=True)
            english_stopwords = stopwords.words('english')
        else:
            english_stopwords = custom_stopwords or []

        self.vectorizer = TfidfVectorizer(stop_words=english_stopwords)

    def fit_transform(self, corpus):
        """
        Fit the vectorizer to the given corpus and transform it into a TF-IDF matrix.

        Parameters:
        - corpus (list): List of text documents.

        Returns:
        - tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF matrix representation of the corpus.
        """
        return self.vectorizer.fit_transform(corpus)

    def transform(self, corpus):
        """
        Transform the given corpus into a TF-IDF matrix using the fitted vectorizer.

        Parameters:
        - corpus (list): List of text documents.

        Returns:
        - tfidf_matrix (scipy.sparse.csr_matrix): TF-IDF matrix representation of the corpus.
        """
        return self.vectorizer.transform(corpus)

    def save_vectorizer(self, filepath):
        """
        Save the fitted vectorizer to a file.

        Parameters:
        - filepath (str): Path to save the vectorizer file.

        Returns:
        None
        """
        dump(self.vectorizer, filepath)

    def load_vectorizer(self, filename):
        """
        Load a saved vectorizer from a file.

        Parameters:
        - filename (str): Name of the vectorizer file to load.

        Returns:
        None
        """
        self.vectorizer = load(filename)