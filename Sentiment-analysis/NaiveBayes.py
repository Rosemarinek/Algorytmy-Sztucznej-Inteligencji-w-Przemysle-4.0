from collections import defaultdict
import pandas as pd
import numpy as np

import Preprocessing


class NaiveBayes:
    def __init__(self, unique_classes):
        # Constructor get unique number of classes of the training set
        self.classes = unique_classes
        self.training_duration = 0

    def add_to_bow(self, text, labels):
        """
        This function splits the text using space as a tokenizer and adss every token to the right bag of words

        :param text: text that should be split
        :param labels: implies to which BoW category this text belongs to
        :return: nothing
        """

        if isinstance(text, np.ndarray):
            text = text[0]
        for token in text.split():
            self.bow[labels][token] += 1

    def train(self, dataset, labels):
        """
        This function trains Naive Bayes Model. Constructs BoW, calculating prior probability and denominator value,
        using smoothing laplace

        :param dataset: dataset which is need to train NB Model
        :param labels: implies to which BoW category this text belongs to
        :return: nothing
        """

        self.dataset_example = dataset
        self.labels = labels
        self.bow = np.array([defaultdict(lambda: 0) for index in range(self.classes.shape[0])])

        if not isinstance(self.dataset_example, np.ndarray):
            self.dataset_example = np.array(self.dataset_example)
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)

        # Constructing BoW
        for index, category in enumerate(self.classes):
            all_category_dataset = self.dataset_example[self.labels == category]  # filter all examples
            cleaned_dataset_example = pd.DataFrame(data=all_category_dataset)

            np.apply_along_axis(self.add_to_bow, 1, cleaned_dataset_example, index)
            probability_classes = np.empty(self.classes.shape[0])
            all_words = []
            category_word_counts = np.empty(self.classes.shape[0])

            for index, category in enumerate(self.classes):
                # Calculating prior probability P(c) for each class
                probability_classes[index] = np.sum(self.labels == category) / float(self.labels.shape[0])

                #Calculating total counts of all the words of each class count(c)
                category_word_counts[index] = np.sum(
                    np.array(list(self.bow[index].values()))) + 1

            # Get all words of this category
            all_words += self.bow[index].keys()

            self.vocabulary = np.unique(np.array(all_words))
            self.vocabulary_length = self.vocabulary.shape[0]

            # Computing denominator value [count(c) + |V|]
            denominator = np.array([category_word_counts[index] + self.vocabulary_length for index, category in
                                    enumerate(self.classes)])

            # Dictionary at index 0, prior probability at index 1, denominator value at index 2
            self.cats_touple = [(self.bow[index], probability_classes[index], denominator[index]) for index, category in
                                enumerate(self.classes)]
            self.cats_touple = np.array(self.cats_touple)

    def getPosteriorProbability(self, example):
        """
        This function estimates posterior probability of the given example

        :param example: example for which this function calculates probability
        :return: probability of example in all classes
        """

        likelihood_probability = np.zeros(self.classes.shape[0])
        posterior_probability = np.empty(self.classes.shape[0])

        for index, category in enumerate(self.classes):
            for token in example.split():  # split the example and get probability of each word
                # Get total count of this token
                token_counts = self.cats_touple[index][0].get(token, 0) + 1
                # Get likelihood of this token
                token_probability = token_counts / float(self.cats_touple[index][2])
                # Using logarithm to prevent underflow
                # log(P(d1|c))+log(P(d2|c))+...log(P(dn|c))
                likelihood_probability[index] += np.log(token_probability)

        # Get posterior probability
        for index, category in enumerate(self.classes):
            # P(c|d1,d2,...,dn) =log(P(d1|c))+log(P(d2|c))+...log(P(dn|c))+log(P(c))
            posterior_probability[index] = likelihood_probability[index] + np.log(self.cats_touple[index][1])

        return posterior_probability

    def test(self, test_set):
        """
        This function calculates probability of each example against all classes and predicts the label against which
        the class probability is maximum

        :param test_set: Set using for testing
        :return: predictions of test examples
        """

        predictions = []
        for example in test_set:
            posterior_probability = self.getPosteriorProbability(example)
            predictions.append(self.classes[np.argmax(posterior_probability)])

        return np.array(predictions)

    def calculate_accuracy(self, test_data, test_labels):
        prediction_classes = self.test(test_data)
        test_accuracy = np.sum(prediction_classes == test_labels) / float(len(test_labels))
        return test_accuracy

    def data_cleaning(self, data):
        data = [Preprocessing.preprocessing_data(data_str) for data_str in data]
        return data

    def predict_sentiment(self, text):
        text = Preprocessing.preprocessing_data(text)
        if(len(text) == 0):
            return ['Neutral']
        prediction = self.test(text)
        return prediction