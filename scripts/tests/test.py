import os
import sys
from sklearn.externals import joblib
from typing import List

# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from multibidaf.common.lem_normalize import lem_normalize
from multibidaf.paths import Paths


def max_cosine_similarity(tfidf_vec, xs, y):
    tfidf_matrix = tfidf_vec.transform([y] + xs)
    cosine_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    return cosine_matrix, cosine_matrix[0, 1:].max()


if __name__ == "__main__":
    # documents = (
    #     "The sky is blue",
    #     "The sun is bright",
    #     "The sun in the sky is bright",
    #     "We can see the shining sun, the bright sun"
    # )
    #
    # tfidf_vectorizer = TfidfVectorizer(documents)
    # train_tfidf_matrix = tfidf_vectorizer.fit(documents)
    #
    # test_tfidf_matrix = tfidf_vectorizer.transform(["The sun is bright", "The sun is not bright"])
    # print((test_tfidf_matrix * test_tfidf_matrix.T).toarray())

    tfidf_vec = joblib.load(Paths.TRAINED_MODELS_ROOT / 'tfidf_vec.pkl')
    xs = ["I was tired", "There is a cat"]
    y = "I am tired"
    cosine_matrix, max_cosine = max_cosine_similarity(tfidf_vec, xs, y)
    print(cosine_matrix)
    print(max_cosine)
