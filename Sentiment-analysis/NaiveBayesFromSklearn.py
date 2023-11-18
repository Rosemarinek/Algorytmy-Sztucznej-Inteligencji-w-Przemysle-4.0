from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import Preprocessing
import numpy as np


class NaiveBayesFromSklearn:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.count_vect = CountVectorizer()
        self.nb = MultinomialNB()
        self.training_duration = 0


    def create_bow_for_train_data(self, train_data):
        X_train_counts = self.count_vect.fit_transform(train_data)
        return X_train_counts

    def train(self):
        X_train_counts = self.create_bow_for_train_data(self.train_data)
        self.nb.fit(X_train_counts, self.train_labels)
        print("---Training Naive Bayes from Sklearn completed---\n")

    def test(self):
        X_test_counts = self.count_vect.transform(self.test_data)
        return X_test_counts

    def calculating_accuracy(self):
        X_test_counts = self.test()
        predicted = self.nb.predict(X_test_counts)
        test_accuracy = np.sum(predicted == self.test_labels) / float(len(predicted))
        return test_accuracy

    def data_cleaning(self, data):
        data = [Preprocessing.preprocessing_data(data_str) for data_str in data]
        return data

    def predict_sentiment(self, text):
        text = self.data_cleaning(text)
        X_text_counts = self.count_vect.transform(text)
        predicted = self.nb.predict(X_text_counts)
        return predicted
