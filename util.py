"""
   utility functions for processing terms

    shared by both indexing and query processing
"""

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def tokenize(text):
    tokens = word_tokenize(text)
    alpha_tokens = [token for token in tokens if token.isalpha()]

    remaining_tokens = [
        # (3) stemming
        stemming(
            token.lower()
            # (1) convert to lower cases
        ) for token in alpha_tokens if
        not isStopWord(token)]  # (2) remove stopwords
    # remaining_tokens=map(lambda s: s.encode("utf-8"), remaining_tokens)

    return remaining_tokens


def isStopWord(word):
    stop_words = [newline.strip() for newline in open("stopwords", "r")]
    if word in stop_words:
        return True
    else:
        return False


""" using the NLTK functions, return true/false"""


def stemming(word):
    stemmer_word = PorterStemmer().stem(word)
    return stemmer_word


""" return the stem, using a NLTK stemmer. check the project description for installing and using it"""
