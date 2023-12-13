import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn import model_selection
from NaiveBayesFromSklearn import *
import dill as pickle
import time
from NaiveBayes import *
from keras.models import load_model
from NeuralNetworks import *



def train_nb_from_sklearn(train_data, train_labels, test_data, test_labels):
    nbS = NaiveBayesFromSklearn(train_data, train_labels, test_data, test_labels)
    print("----Training with an algorithm from Sklearn in progress----")
    st = time.time()
    nbS.train()
    et = time.time()
    nbS.training_duration = et - st
    print("Training NBSklearn duration: ", nbS.training_duration)
    print('----Training Completed----\n')

    fS = open('nbSklearn_classifier.pickle', 'wb')
    pickle.dump(nbS, fS)
    fS.close()

def train_nb(train_data, train_labels, classes):
    nb = NaiveBayes(classes)
    print("----Training with an implemented algorithm in progress---")
    st = time.time()
    nb.train(train_data, train_labels)
    et = time.time()
    nb.training_duration = et - st
    print("Training NB duration: ", nb.training_duration)
    print('---Training Completed---\n')

    f = open('nb_classifier.pickle', 'wb')
    pickle.dump(nb, f)
    f.close()

def train_bilstm(train_data, train_labels, test_data, test_labels):
    bilstm = BiLSTM(train_data, train_labels, test_data, test_labels)
    print("----Training BiLSTM in progress---")
    st = time.time()
    bilstm.train()
    et = time.time()
    bilstm.training_duration = et - st
    print("Training NB duration: ", bilstm.training_duration)
    print('---Training Completed---\n')

    f = open('bilstm.pickle', 'wb')
    pickle.dump(bilstm, f)
    f.close()



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # ------------------------- Upload data -----------------------------#

    print("------------------------- Upload data -------------------------")
    training_set = pd.read_csv('./data/IMDBDataset.csv', header=None, sep=',')
    x_train = training_set[0].values[1:]  # content, delete headers
    y_train = training_set[1].values[1:]  # sentiment, delete headers

    # -------------------------------------------------------------------#

    # ------------Plot The total number of examples in dataset----------#

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

    # -------------------------------------------------------------------#

    # ------------------------Preprocessing data-------------------------#

    print("----------------------- Preprocessing data -----------------------")
    x_train_after_preprocessing = []
    y_train_after_preprocessing = []
    for index in range(len(x_train)):
        text = Preprocessing.preprocessing_data(x_train[index])
        if text != "":
            x_train_after_preprocessing.append(text)
            y_train_after_preprocessing.append(y_train[index])

    pd.DataFrame(x_train_after_preprocessing).to_csv("x_train.csv", header=None, sep=";")
    pd.DataFrame(y_train_after_preprocessing).to_csv("y_train.csv", header=None, sep=";")

    # -------------------------------------------------------------------#

    # -------------Upload data after preprocessing data------------------#

    print("------------------- Upload data after preprocessing data --------------------")
    x_train_after_preprocessing = pd.read_csv('x_train.csv', header=None, sep=";")[1].values
    y_train_after_preprocessing = pd.read_csv('y_train.csv', header=None, sep=";")[1].values

    # -------------------------------------------------------------------#

    # ------Plot The total number of examples after preprocessing--------#

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

    # -------------------------------------------------------------------#

    # ---------------------------Split data------------------------------#
    train_data, test_data, train_labels, test_labels = model_selection.train_test_split(x_train_after_preprocessing,
                                                                                        y_train_after_preprocessing,
                                                                                        shuffle=True,
                                                                                        test_size=0.25,
                                                                                        random_state=42,
                                                                                        stratify=y_train_after_preprocessing)
    classes = np.unique(train_labels)

    # -------------------------------------------------------------------#

    # ------Plot The total number of training and test examples----------#

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

    # --------------------------------------------------------------------#

    # -----------------------------TRAINING------------------------------#
    # Training NB from Sklearn
    train_nb_from_sklearn(train_data, train_labels, test_data, test_labels)

    # Training NB own algorithm
    train_nb(train_data, train_labels, classes)


    # Training BiLSTM
    train_bilstm(train_data, train_labels, test_data, test_labels)

    # --------------------------------------------------------------------#

    # ---------------------------OPEN MODELS------------------------------#
    fS = open('nbSklearn_classifier.pickle', 'rb')
    nbS = pickle.load(fS)
    fS.close()

    f = open('nb_classifier.pickle', 'rb')
    nb = pickle.load(f)
    f.close()


    bilstm_model = load_model('bilstm.h5')
    f = open('bilstm.pickle', 'rb')
    bilstm = pickle.load(f)
    f.close()

    print(bilstm.accuracy)

    # --------------------------------------------------------------------#

    # ----------------CALCULATING AND PLOT ACCURRACY -----------------------#
    print("----------------Calculating accurracy-----------------------")
    nbS_accuracy = nbS.calculate_accuracy()
    nb_accuracy = nb.calculate_accuracy(test_data, test_labels)

    plt.figure(5)
    fig2, ax2 = plt.subplots()
    plt.grid(color='#CCCCCC', linestyle=':', linewidth=1)
    height = [nb_accuracy * 100, nbS_accuracy * 100, bilstm.accuracy * 100]
    y_pos = np.arange(len(height))
    bars = ["Stworzony algorytm", "Algorytm z biblioteki Sklearn", "BiLSTM"]
    plt.bar(y_pos, height, color=['forestgreen', 'cornflowerblue', 'orange'])
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

    # --------------------------------------------------------------------#

    # ---------------------PLOT TRAINING DURATION-------------------------#

    plt.figure(6)
    fig2, ax2 = plt.subplots()
    plt.grid(color='#CCCCCC', linestyle=':', linewidth=1)
    height = [nb.training_duration, nbS.training_duration, bilstm.training_duration]
    y_pos = np.arange(len(height))
    bars = ["Stworzony algorytm", "Algorytm z biblioteki \n Sklearn", "BiLSTM"]
    plt.bar(y_pos, height, color=['lawngreen', 'dodgerblue', 'orange'])
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



