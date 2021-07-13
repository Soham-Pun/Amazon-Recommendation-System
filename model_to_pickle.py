import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import joblib


data = pd.read_pickle('Dataset1')
data.drop(['index'], axis=1, inplace=True)
data.reset_index(inplace=True, drop=True)

model_wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=1500000)
vocab = model_wv.index_to_key

def create_tfidf_vector():
    tfidf_vectorizer = TfidfVectorizer(min_df=0)
    tfidf_word_matrix = tfidf_vectorizer.fit_transform(data['title'])
    joblib.dump(tfidf_word_matrix, 'tfidf_word_matrix.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')




def build_wordvec(text, doc_id, m_name):

    featureVector = np.zeros((300,), dtype="float32")
    numwords = 0
    tfidf_title_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    tfidf_title_matrix = joblib.load('tfidf_word_matrix.pkl')
    for word in text.split():
        numwords += 1
        if word in vocab:
            if m_name == 'weighted' and word in tfidf_title_vectorizer.vocabulary_:
                featureVector = np.add(featureVector,
                                    tfidf_title_matrix[doc_id, tfidf_title_vectorizer.vocabulary_[word]] * model_wv[word])
            elif m_name == 'avg':
                featureVector = np.add(featureVector, model_wv[word])
    if (numwords > 0):
        featureVector = np.divide(featureVector, numwords)
    # returns the avg vector of given sentance, its of shape (1, 300)
    return featureVector


def tfidf_wordvec_model():
    doc_id = 0
    w2v_title_weight = []
    for i in data['title']:
        w2v_title_weight.append(build_wordvec(i, doc_id, 'weighted'))
        doc_id += 1

    w2v_title_weight = np.array(w2v_title_weight)
    joblib.dump(w2v_title_weight, 'wordvec_model.pkl')

create_tfidf_vector()
tfidf_wordvec_model()