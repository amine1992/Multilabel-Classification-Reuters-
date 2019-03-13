
import sys
import re
import numpy as np
import nltk

nltk.download('stopwords')
nltk.download('reuters')

from nltk.corpus import reuters
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

def w2v(docs, size_vects):
    w2v_model = Word2Vec(docs,
                        size=size_vects,
                        min_count=1,
                        window=10,
                        workers=cpu_count())
    w2v_model.init_sims(replace=True)
    w2v_model.save('./reuters_train.word2vec')
    return w2v_model

def w2v_vectorize_docs(documents, number_of_documents, w2v_model, config):
    """A weird oneshot representation for word2vec."""


    document_max_num_words = config.getint('DATA', 'document_max_num_words')
    num_features = config.getint('DATA', 'num_features')

    x = np.zeros(shape=(number_of_documents, document_max_num_words,
                        num_features)).astype(np.float32)
    empty_word = np.zeros(num_features).astype(np.float32)

    for idx, document in enumerate(documents):
        for jdx, word in enumerate(document):
            if jdx == document_max_num_words:
                break

            else:
                if word in w2v_model:
                    x[idx, jdx, :] = w2v_model[word]
                else:
                    x[idx, jdx, :] = empty_word

    return x

def save_data(x_train, y_train, x_test, y_test):
    np.save('./data/x_train_np.npy', x_train)
    np.save('./data/y_train_np.npy', y_train)
    np.save('./data/x_test_np.npy', x_test)
    np.save('./data/y_test_np.npy', y_test)

n_classes = 90
labels = reuters.categories()

def read_retuters_files():
    """
    Load the Reuters dataset.
    Returns
    -------
    data : dict
        with keys 'x_train', 'x_test', 'y_train', 'y_test', 'labels'
    """
    stop_words = stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    mlb = MultiLabelBinarizer()

    documents = reuters.fileids()
    test = [d for d in documents if d.startswith('test/')]
    train = [d for d in documents if d.startswith('training/')]

    docs = {}
    docs['train'] = [reuters.raw(doc_id) for doc_id in train]
    docs['test'] = [reuters.raw(doc_id) for doc_id in test]

    ys = {'train': [], 'test': []}
    ys['train'] = mlb.fit_transform([reuters.categories(doc_id)
                                     for doc_id in train])
    ys['test'] = mlb.transform([reuters.categories(doc_id)
                                for doc_id in test])
    
    data = {'x_train': docs['train'], 'y_train': ys['train'],
            'x_test': docs['test'], 'y_test': ys['test'],
            'labels': globals()["labels"]}

    # save_data(data['x_train'], data['y_train'], data['x_test'], data['y_test'])

    return (data['x_train'], data['y_train']), (data['x_test'], data['y_test'])

