import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load

class TextVectorizer:
    def __init__(self, use_default_stopwords=True, custom_stopwords=None, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize the TextVectorizer class.

        Parameters:
        - use_default_stopwords (bool): Flag to use default English stopwords.
        - custom_stopwords (list): Custom stopwords, if any.
        - max_features (int): Maximum number of features for the vectorizer.
        - ngram_range (tuple): Range of n-values for n-grams.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.max_features = max_features
        self.ngram_range = ngram_range

        self.stopwords = self._get_stopwords(use_default_stopwords, custom_stopwords)
        self.vectorizer = self._initialize_vectorizer()

    def _get_stopwords(self, use_default, custom):
        """
        Retrieve stopwords based on the user's choice.

        Parameters:
        - use_default (bool): Flag to use default English stopwords.
        - custom (list): Custom stopwords, if any.

        Returns:
        - list: List of stopwords.
        """
        if use_default:
            nltk.download('stopwords', quiet=True)
            return stopwords.words('english')
        else:
            return custom or []

    def _initialize_vectorizer(self):
        """
        Initialize the TfidfVectorizer with the specified settings.

        Returns:
        - TfidfVectorizer: Initialized TfidfVectorizer object.
        """
        nltk.download(['wordnet', 'omw-1.4'], quiet=True)
        return TfidfVectorizer(
            stop_words=self.stopwords,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            norm='l2'
        )

    def _preprocess(self, text):
        """
        Preprocess text by removing special characters and lemmatizing.

        Parameters:
        - text (str): Input text to preprocess.

        Returns:
        - str: Preprocessed text.
        """
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])

    def fit_transform(self, corpus):
        """
        Fit and transform the corpus into a TF-IDF matrix.

        Parameters:
        - corpus (list): List of texts to fit and transform.

        Returns:
        - scipy.sparse.csr_matrix: TF-IDF matrix representation of the corpus.
        """
        preprocessed_corpus = [self._preprocess(text) for text in corpus]
        return self.vectorizer.fit_transform(preprocessed_corpus)

    def transform(self, corpus):
        """
        Transform the corpus into a TF-IDF matrix using the fitted vectorizer.

        Parameters:
        - corpus (list): List of texts to transform.

        Returns:
        - scipy.sparse.csr_matrix: TF-IDF matrix representation of the corpus.
        """
        preprocessed_corpus = [self._preprocess(text) for text in corpus]
        return self.vectorizer.transform(preprocessed_corpus)

    def save_vectorizer(self, filepath):
        """
        Save the fitted vectorizer to a file.

        Parameters:
        - filepath (str): Filepath to save the vectorizer.
        """
        dump(self.vectorizer, filepath)

    def load_vectorizer(self, filename):
        """
        Load a saved vectorizer from a file.

        Parameters:
        - filename (str): Filename of the saved vectorizer.
        """
        self.vectorizer = load(filename)
