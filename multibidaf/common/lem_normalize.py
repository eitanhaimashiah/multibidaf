import nltk
import string

nltk.download('wordnet')  # first-time use only
lemmer = nltk.stem.WordNetLemmatizer()


def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def lem_normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
