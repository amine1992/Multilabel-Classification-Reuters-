"""A simple MLP to run the classification."""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from metrics import precision, recall, f1
from keras.models import load_model

def mlp(X_train, Y_train, X_test, Y_test, config):
    document_max_num_words = config.getint('DATA', 'document_max_num_words')
    num_features = config.getint('DATA', 'num_features')
    num_classes = config.getint('DATA', 'num_classes')
    batch_size = config.getint('MODEL', 'batch_size')
    epochs = config.getint('MODEL', 'epochs')

    print(num_classes, 'classes')

    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)

    print('y_train shape:', Y_train.shape)
    print('y_test shape:', Y_test.shape)

    print('Building model...')
    model = Sequential()
    if len(X_train.shape)==3:
        model.add(Flatten(input_shape=(document_max_num_words, num_features)))
        model.add(Dense(512))
    else:
        model.add(Dense(512, input_shape=(X_train.shape[1],)))
    
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy", precision, recall, f1])

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.1)

    model.save('mlp_reuters.h5', overwrite=True)
    
    # model = load_model('mlp_reuters.h5')
    
    score = model.evaluate(X_test, Y_test,
                           batch_size=batch_size, verbose=1)

    print('test score: %1.4f' % score[0])
    print('test accuracy: %1.4f' % score[1])
    print('test precision: %1.4f' % score[2])
    print('test recall: %1.4f' % score[3])
    print('test f1: %1.4f' % score[4])

