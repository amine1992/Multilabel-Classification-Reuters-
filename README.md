# Topic Classification for Reuters-21578 dataset

## Train model 

The idea is to run a classification task based on the reuters-21578 dataset.

In order to run the project, fist install all the pip requirements:

```bash
pip install -r requirements.txt
```

To run the training functions:

```
python train.py --config_file run.conf

```

The run.conf is a configuration file to run the training divided by section.

### DATA

* num_doc_train: Number of documents in the training.
* num_doc_test: number of documents in the test.
* document_max_num_words: maximum words in a document.
* num_features: the size of the embedding of the words (using word2vec).
* num_classes: number of classes in the multiclassification.

### Model

* log_dir: directory to save the log
* model: the model to train our data on. can be: mlp (multilayer perceptron) or lstm.
vectorization: how to vectorize the documents. can be tfidf or w2v


## Analysis:

### Models

I tried to use 2 different deep models:

* LSTM : one of the most succesful deep architecture for sequential data, obviously the first model to try when performing NLP. It tries to go throught the document, prepare a kind of summary then suggest classes. The performance on this task was not so promising comparing to MLP. The reasons can be: Data is small for a model like LSTM, then, the size of the documents is long to remember the pieces that define the classes to predict.
* MLP: multilayer perceptron: with a very simple vectorization using TFIDF and 2 layers, it could achieve a decent performance compared to the mlp. The reason is that the task itself relies on the presence of key words that can refer to certain classes. So an appropriate representation highlighing those words and a small non linear transformation to combine the features can do the thing.

#### Classification with Reject option

For a problem like multilabel classification, it is not straightforward to get a reject option as a postprocessing. 
The design I have in mind to solve it is the following:
1. LSTM : we will train an LSTM architecture to solve the task. We wouldn't care if it overfits a little bit. At the end, all we care about is the context vectors (hidden state of the final timestep) of each document in the training. 
2. Clustering of the context vectors: once the LSTM model is ready and per document, we have the context vector. We will perform a clustering on these vectors. We can set the number of clusters to be little bit high. At the end of this step, we will have the training documents grouped.
3. IQR for outlier detection: Now, for each new document, we will feed it to the LSTM, get the context vector, assign it to a cluster (compare the distance with each cluster center and assign it to the nearest). Finally, we The decision of rejecting or not will be treated as an outlier detection depending on the distance (example, assigned_cluster) and the cluster points: If the distance is less Lower inner fence: Q1 – (1.5 * IQR) or more than Upper fence = Q3 + (1.5 * IQR), we'll reject the example. 





