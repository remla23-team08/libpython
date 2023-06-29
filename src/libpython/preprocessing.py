"""
This file contains functions related to preprocessing of any data provided to the model
"""

import re

import nltk  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.stem.porter import PorterStemmer  # type: ignore


class Preprocessing:
    """Class to easily preprocess datasets"""

    def __init__(self):
        """Initialize preprocess class"""

        nltk.download("stopwords")
        self.porter_stem = PorterStemmer()
        self.all_stopwords = stopwords.words("english")
        self.all_stopwords.remove("not")

        self.dataset = None
        self.count_vectorizer = None

    def preprocess_dataset(self, dataset):
        """Loop over entire dataset to preprocess"""
        corpus = []
        for i in range(0, len(dataset)):
            corpus.append(self.preprocess_review(dataset["Review"][i]))
        return corpus

    def preprocess_review(self, review):
        """Processing a single review"""

        review = re.sub("[^a-zA-Z]", " ", review)
        review = review.lower()
        review = review.split()
        review = [
            self.porter_stem.stem(word)
            for word in review
            if not word in set(self.all_stopwords)
        ]
        review = " ".join(review)
        return review
