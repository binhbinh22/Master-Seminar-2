"""TF-IDF feature extraction for the code-comment multi-label classifier.

Word n-grams capture vocabulary/phrasing, char n-grams capture sub-word and morphological
patterns (useful since comments are short and full of identifier-like tokens). Both are
fit fresh inside each CV fold (see evaluation.py) so no vocabulary derived from a fold's
held-out text leaks into training for that fold.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_vectorizer():
    word_tfidf = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', min_df=2, max_features=20000)
    char_tfidf = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', min_df=2, max_features=20000)
    return FeatureUnion([('word', word_tfidf), ('char', char_tfidf)])
