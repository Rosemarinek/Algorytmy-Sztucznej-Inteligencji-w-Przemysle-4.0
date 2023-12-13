import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder



class BiLSTM:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.training_duration = 0
        self.accuracy = 0

        #Parameters
        self.max_words = 10000
        self.maxlen = 100
        self.embedding_dim = 50
        self.neurons_bilstm = 128
        self.dropout = 0.5
        self.neurons_dense = 1
        self.activation_function = 'sigmoid'
        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'
        self.metrics = ['accuracy']
        self.epochs = 1
        self.batch_size = 32


    def train(self):
        # Tokenizacja tekstu
        tokenizer = Tokenizer(num_words=self.max_words)
        tokenizer.fit_on_texts(self.train_data)
        train_sequences = tokenizer.texts_to_sequences(self.train_data)
        test_sequences = tokenizer.texts_to_sequences(self.test_data)

        # Ustawienie stałej długości sekwencji
        self.train_data = pad_sequences(train_sequences, maxlen=self.maxlen)
        self.test_data = pad_sequences(test_sequences, maxlen=self.maxlen)

        label_encoder = LabelEncoder()
        train_labels_numeric = label_encoder.fit_transform(self.train_labels)
        test_labels_numeric = label_encoder.fit_transform(self.test_labels)

        # Budowa modelu Bi-LSTM
        model = Sequential()
        model.add(Embedding(input_dim=self.max_words, output_dim=self.embedding_dim, input_length=self.maxlen))
        model.add(Bidirectional(LSTM(self.neurons_bilstm)))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.neurons_dense, activation=self.activation_function))

        # Kompilacja modelu
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        # Uczenie modelu
        model.fit(self.train_data, train_labels_numeric, epochs=self.epochs, batch_size=self.batch_size,
                  validation_data=(self.test_data, test_labels_numeric))

        # Predykcja na zbiorze testowym
        predictions = model.predict(self.test_data)
        predictions_binary = (predictions > 0.5).astype(int)

        # Wyliczenie dokładności
        self.accuracy = accuracy_score(test_labels_numeric, predictions_binary)

        print("Accuracy: {:.2f}%".format(self.accuracy))
        model.save('bilstm.h5')

