from flask import Flask, request, url_for
import flask
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils import shuffle
import joblib


app = Flask(__name__)

data = pd.read_pickle('Dataset1')
data.drop(['index'], axis=1, inplace=True)
data.reset_index(inplace=True, drop=True)

@app.route('/')
def display_product():
    # sample = shuffle(data).head(50)
    sample = data.head(50)
    return flask.render_template('index.html', dataset=sample)

@app.route('/similarProducts')
def similarProduct():
    asin = request.args.get('asin', type=str)
    num_of_prod = 20
    index = data[data['asin'] == asin].index[0]
    query_prod = data.loc[index][:]
    avg_wordvec_results = tfidf_wordvec(index, num_of_prod)
    index_results = list(avg_wordvec_results)

    results = data.loc[index_results][:]
    results.reset_index(inplace=True, drop=True)
    return flask.render_template('SimilarProducts.html', result=results, query=query_prod)




def tfidf_wordvec(index, num_of_prod):
    avg_wordvec_matrix = joblib.load('wordvec_model.pkl')
    pairwise_dist = pairwise_distances(avg_wordvec_matrix, avg_wordvec_matrix[index].reshape(1, -1))

    # np.argsort will return indices of 9 smallest distances
    sorted_indices_of_distance = np.argsort(pairwise_dist.flatten())[0:num_of_prod]
    return sorted_indices_of_distance


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)