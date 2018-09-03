import os
import sys
import nltk
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from multibidaf.common.lem_normalize import lem_normalize
from multibidaf.dataset_readers import MultiRCDatasetReader
from multibidaf.paths import Paths

nltk.download('punkt')

if __name__ == "__main__":
    reader = MultiRCDatasetReader(lazy=False)
    reader.read(Paths.DATA_ROOT / "multirc_train.json")
    documents = reader.get_documents()

    tfidf_vec = TfidfVectorizer(tokenizer=lem_normalize, stop_words='english')
    tfidf_vec.fit(documents)
    joblib.dump(tfidf_vec, Paths.TRAINED_MODELS_ROOT / 'tfidf_vec.pkl')
