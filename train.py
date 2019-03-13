import numpy as np

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
import os
from multiprocessing import cpu_count

import configparser
import argparse

import lstm
import mlp
import prep_data


# this = sys.modules[__name__]


def read_numpy_files():
    
    """Instead of running the entire pipeline at all times."""
    x_train = np.load('./data/x_train_np.npy')
    y_train = np.load('./data/y_train_np.npy')

    x_test = np.load('./data/x_test_np.npy')
    y_test = np.load('./data/y_test_np.npy')

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Reuters')
    parser.add_argument("--config_file", dest="config_file",\
                    action="store", default="run.conf",\
                    required=False, help="The config file to store the configuration")
    
    args = parser.parse_args()
    
    config_file = args.config_file
    config = configparser.ConfigParser()
    config.read(config_file)
    
    (x_train, y_train), (x_test, y_test) = prep_data.read_retuters_files()
    
    num_doc_train = config.getint('DATA', 'num_doc_train')
    num_doc_test = config.getint('DATA', 'num_doc_test')
    num_features = config.getint('DATA', 'num_features')

    if config.get('MODEL', 'vectorization') == "Word2Vec":
        # Vectorize train
        w2v_model = prep_data.w2v(x_train, num_features)
        x_train = prep_data.vectorize_docs(x_train, num_doc_train, w2v_model, config)
        # Vectorize test
        w2v_model = prep_data.w2v(x_test, num_features)
        x_test = prep_data.vectorize_docs(x_test, num_doc_test, w2v_model, config)
    
    elif config.get('MODEL', 'vectorization') == "tfidf":
        stop_words = stopwords.words("english")
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        x_train = vectorizer.fit_transform(x_train).toarray()
        x_test = vectorizer.transform(x_test).toarray()
    
    log_dir = os.path.join(os.getcwd(), config.get('MODEL', 'log_dir'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Check which model to run
    if config.get('MODEL', 'model') == 'lstm':
        lstm.lstm(x_train, y_train, x_test, y_test, config)
    elif config.get('MODEL', 'model') == 'mlp':
        mlp.mlp(x_train, y_train, x_test, y_test, config)
    else:
        print (" error with model, choose between 'mlp' and 'lstm' ")

