import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class TextVectorizer:
    """
    TextVectorizer is a class for converting a collection of text documents to a matrix of token counts
    and transforming this count matrix to a tf-idf representation.

    Attributes:
        count_vect (CountVectorizer): Instance of CountVectorizer.
        tfidf_transformer (TfidfTransformer): Instance of TfidfTransformer.
    """

    def __init__(self):
        """
        Initializes the TextVectorizer with CountVectorizer and TfidfTransformer instances.
        """
        self.count_vect = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()

    def fit_transform(self, corpus):
        """
        Fits the vectorizer to the provided corpus and transforms it into tf-idf representation.

        Args:
            corpus (list of str): A list of text documents.

        Returns:
            scipy.sparse.csr.csr_matrix: The tf-idf representation of the corpus.
        """
        X_train_counts = self.count_vect.fit_transform(corpus)
        X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)
        return X_train_tfidf

    def transform(self, new_corpus):
        """
        Transforms a new corpus into the tf-idf representation based on the fitted vectorizer.

        Args:
            new_corpus (list of str): A new list of text documents.

        Returns:
            scipy.sparse.csr.csr_matrix: The tf-idf representation of the new corpus.
        """
        X_new_counts = self.count_vect.transform(new_corpus)
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
        return X_new_tfidf

    def save(self, filepath):
        """
        Saves the state of the vectorizer to a file.

        Args:
            filepath (str): The path where the vectorizer state will be saved.
        """
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filepath):
        """
        Loads a TextVectorizer from a file.

        Args:
            filepath (str): The path to the file containing the saved state.

        Returns:
            TextVectorizer: The loaded TextVectorizer object.
        """
        with open(filepath, 'rb') as file:
            return pickle.load(file)
