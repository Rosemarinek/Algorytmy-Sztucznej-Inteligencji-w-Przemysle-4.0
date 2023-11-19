import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.casual import casual_tokenize


def lemmatisation(text):
    """
    This function is responsible for lemmatisation. Find the root form of words or lemmas in NLP and remove stop words.
    :param text:
    :return: String after lemmatisation
    """

    stop_words = set(stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatize_data = ""

    tokens = casual_tokenize(str(text), reduce_len=True, strip_handles=True)
    for word in tokens:
        if not word in stop_words and len(word) > 0:
            lemmatize_data = lemmatize_data + lemmatizer.lemmatize(word) + " "
    return lemmatize_data


def preprocessing_data(text):
    """
    This function preprocess the text:
    1. everything apart from letters is excluded
    2. multiple spaces are replaced by single space
    3. data is converted to lower case
    4. find the root form of words or lemmas in NLP and remove stop words

    :param text:  text that should be preprocessed
    :return: Preprocessed string
    """
    preprocess_data = re.sub(r'http\S+', ' ', str(text), flags=re.IGNORECASE) #delete http/https
    preprocess_data = re.sub(r'(^| )\S+\.com( |$)', ' ', preprocess_data, flags=re.IGNORECASE) #delete link.com
    preprocess_data = re.sub(r'<\S+\W?/?>', ' ', preprocess_data) #delete html marks e.g <br><br/>
    preprocess_data = re.sub(r'(@\w+|\d+)', ' ', preprocess_data, flags=re.IGNORECASE) #delete user names
    preprocess_data = re.sub(r'[^a-z\s]+', ' ', preprocess_data, flags=re.IGNORECASE) #leave only letters and spaces
    preprocess_data = re.sub(r'(\s+)', ' ', preprocess_data) #delete unnecesary spaces
    preprocess_data = preprocess_data.lower()
    preprocess_data = lemmatisation(preprocess_data)
    preprocess_data = re.sub(r'[^a-z\s]+', ' ', preprocess_data, flags=re.IGNORECASE) #leave only letters and spaces
    preprocess_data = re.sub(r'(^| ).( |$)', '', preprocess_data) #delete individual letters
    preprocess_data = re.sub(r'(\s+)', ' ', preprocess_data) #delete unnecesary spaces

    return preprocess_data


def data_cleaning(data):
    data = [preprocessing_data(data_str) for data_str in data]
    return data
