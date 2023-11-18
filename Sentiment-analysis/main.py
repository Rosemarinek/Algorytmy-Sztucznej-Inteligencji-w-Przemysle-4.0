import pandas as pd
import warnings
import matplotlib.pyplot as plt
import Preprocessing
from sklearn import model_selection
import numpy as np
from NaiveBayesFromSklearn import *
import dill as pickle

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # ------------------------- Upload data -----------------------------#

    print("------------------------- Upload data -------------------------")
    training_set = pd.read_csv('./data/IMDBDataset.csv', header=None, sep=',')
    x_train = training_set[0].values[1:]  # content, delete headers
    y_train = training_set[1].values[1:]  # sentiment, delete headers

    # # -------------------------------------------------------------------#
    #
    # # ------------Plot The total number of examples in dataset----------#

    plt.figure(1)
    fig1, ax1 = plt.subplots()
    N, bins, patches = ax1.hist(y_train, bins=3)
    patches[0].set_facecolor('green')
    patches[2].set_facecolor('red')
    plt.grid(color='#CCCCCC', linestyle=':', linewidth=1)
    plt.title('Liczba przykładów w zbiorze danych dla każdej kategorii sentymentu')
    plt.xlabel("Kategoria sentymentu", fontsize=10)
    plt.ylabel("Liczba przykładów", fontsize=10)
    plt.savefig("images/1.png")
    plt.show()

    # # -------------------------------------------------------------------#
    #
    # # ------------------------Preprocessing data-------------------------#

    # print("----------------------- Preprocessing data -----------------------")
    x_train_after_preprocessing = []
    y_train_after_preprocessing = []
    for index in range(len(x_train)):
        text = Preprocessing.preprocessing_data(x_train[index])
        if text != "":
            x_train_after_preprocessing.append(text)
            y_train_after_preprocessing.append(y_train[index])

    pd.DataFrame(x_train_after_preprocessing).to_csv("x_train.csv", header=None, sep=";")
    pd.DataFrame(y_train_after_preprocessing).to_csv("y_train.csv", header=None, sep=";")

    # # -------------------------------------------------------------------#
    #
    # # -------------Upload data after preprocessing data------------------#
    #
    print("------------------- Upload data after preprocessing data --------------------")
    x_train_after_preprocessing = pd.read_csv('x_train.csv', header=None, sep=";")[1].values
    y_train_after_preprocessing = pd.read_csv('y_train.csv', header=None, sep=";")[1].values

    # # -------------------------------------------------------------------#
    #
    # # ------Plot The total number of examples after preprocessing--------#

    plt.figure(2)
    fig3, ax3 = plt.subplots()
    N3, bins3, patches3 = ax3.hist([y_train, y_train_after_preprocessing],
                                   label=['Przed przetworzeniem', 'Po przetworzeniu'], bins=3)
    plt.legend(loc='upper right')
    plt.grid(color='#CCCCCC', linestyle=':', linewidth=1)
    plt.title('Liczba przykładów w zbiorze danych dla każdej kategorii sentymentu \n przed i po przetworzeniu danych')
    plt.xlabel("Kategoria sentymentu", fontsize=10)
    plt.ylabel("Liczba przykładów", fontsize=10)
    plt.savefig("images/2.png")
    plt.show()

    # # -------------------------------------------------------------------#
    #
    # # ---------------------------Split data------------------------------#
    train_data, test_data, train_labels, test_labels = model_selection.train_test_split(x_train_after_preprocessing,
                                                                                        y_train_after_preprocessing,
                                                                                        shuffle=True,
                                                                                        test_size=0.25,
                                                                                        random_state=42,
                                                                                        stratify=y_train_after_preprocessing)
    classes = np.unique(train_labels)

    # # -------------------------------------------------------------------#
    #
    # # ------Plot The total number of training and test examples----------#

    plt.figure(3)
    fig24, ax4 = plt.subplots()
    plt.grid(color='#CCCCCC', linestyle=':', linewidth=1)
    height = [len(train_data), len(test_data)]
    y_pos = np.arange(len(height))
    bars = ["Dane treningowe", "Dane testowe"]
    plt.bar(y_pos, height, color=['dodgerblue', 'darkorange'])
    plt.xticks(y_pos, bars)
    plt.title('Liczba przykładów treningowych i testowych')
    plt.xlabel("Rodzaj danych", fontsize=10)
    plt.ylabel("Liczba przykładów", fontsize=10)
    rects = ax4.patches
    des = ["75%", "25%"]
    i = 0
    for rect, height in zip(rects, height):
        height_ann = rect.get_height()
        ax4.text(
            rect.get_x() + rect.get_width() / 2, height_ann, des[i], ha="center", va="bottom"
        )
        i = i + 1
    plt.savefig("images/3.png")
    plt.show()

    # # --------------------------------------------------------------------#
    #
    # # -----------------------------TRAINING------------------------------#

    # Training NB from Sklearn
    train_nb_from_sklearn(train_data, train_labels, test_data, test_labels)

    # # --------------------------------------------------------------------#
    #
    # # ---------------------------OPEN MODELS------------------------------#
    fS = open('nbSklearn_classifier.pickle', 'rb')
    nbS = pickle.load(fS)
    fS.close()

    # # --------------------------------------------------------------------#
    #
    # # ----------------CALCULATING AND PLOT ACCURRACY -----------------------#
    print("----------------Calculating accurracy-----------------------")
    nbS_accuracy = nbS.calculating_accuracy()

    plt.figure(5)
    fig2, ax2 = plt.subplots()
    plt.grid(color='#CCCCCC', linestyle=':', linewidth=1)
    height = [nbS_accuracy * 100]
    y_pos = np.arange(len(height))
    bars = ["Algorytm z biblioteki Sklearn"]
    plt.bar(y_pos, height, color=['cornflowerblue'])
    plt.xticks(y_pos, bars)
    plt.yticks(np.arange(0, 110, 10))
    plt.xlabel("Rodzaj algorytmu", fontsize=10)
    plt.ylabel("Dokładność [%]", fontsize=10)
    rects = ax2.patches
    for rect, height in zip(rects, height):
        height_ann = rect.get_height()
        ax2.text(
            rect.get_x() + rect.get_width() / 2, height_ann, height, ha="center", va="bottom"
        )
    plt.savefig("images/accuracy.png")
    plt.show()

    # # --------------------------------------------------------------------#
    #
    # # ---------------------PLOT TRAINING DURATION-------------------------#

    plt.figure(6)
    fig2, ax2 = plt.subplots()
    plt.grid(color='#CCCCCC', linestyle=':', linewidth=1)
    height = [nbS.training_duration]
    y_pos = np.arange(len(height))
    bars = ["Algorytm z biblioteki \n Sklearn"]
    plt.bar(y_pos, height, color=['dodgerblue'])
    plt.xticks(y_pos, bars)
    plt.xlabel("Rodzaj algorytmu", fontsize=10)
    plt.ylabel("Czas [s]", fontsize=10)
    rects = ax2.patches
    for rect, height in zip(rects, height):
        height_ann = rect.get_height()
        ax2.text(
            rect.get_x() + rect.get_width() / 2, height_ann, height, ha="center", va="bottom"
        )
    plt.savefig("images/duration.png")
    plt.show()



