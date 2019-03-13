import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
import sys
from metrics import precision, recall, f1

def lstm(X_train, Y_train, X_test, Y_test, config):
    """Create the LSTM model."""
    document_max_num_words = config.getint('DATA', 'document_max_num_words')
    num_features = config.getint('DATA', 'num_features')
    num_classes = config.getint('DATA', 'num_classes')
    batch_size = config.getint('MODEL', 'batch_size')
    epochs = config.getint('MODEL', 'epochs')

    tb_callback = keras.callbacks.TensorBoard(log_dir='./tb', histogram_freq=0,
                                              write_graph=True, write_images=True)

    model = Sequential()
    model.add(LSTM(int(document_max_num_words * 1.5), 
        input_shape=(document_max_num_words, num_features)))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', 
        metrics=['accuracy', precision, recall, f1])

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(X_test, Y_test), callbacks=[tb_callback])

    model.save('lstm_reuters.h5', overwrite=True)

    score = model.evaluate(X_test, Y_test, batch_size=batch_size)

    print('test score: %1.4f' % score[0])
    print('test accuracy: %1.4f' % score[1])
    print('test precision: %1.4f' % score[2])
    print('test recall: %1.4f' % score[3])
    print('test f1: %1.4f' % score[4])
