import pickle

import nltk
import numpy as np
import pandas as pd
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


def find_senteces_with_lemma(lemma, tokens, sentences, sentences_orig, X, no_of_examples=10, PRINT=True):
    ''' search in all sentences in {sentences} for the {lemma}
        print the original sentencess from which thie lemma was extracted
        and return list of those sentences'''

    # for example:
    # find_senteces_with_lemma("we are", tokens_005_no_digits,headlines_train, headlines_train_orig,X_train,PRINT = True)
    lemma_index = tokens.index(lemma)
    indices = (X[:, lemma_index].nonzero())[0]
    if PRINT:
        print("\t-----------------------")
        print(f"\tThere are {len(indices)} sentences containing the lemma - '{lemma}'\n" +
              f"\tThis equal to {round(len(indices) / len(sentences) * 100, 2)}% of total data")
        print("\t-----------------------")

    r = []
    for i in indices[:no_of_examples]:
        s = sentences_orig[i]
        if PRINT: print(s)
        r.append(s)
    return r


def accuracy(Y_pred, Y_true):
    #     print (Y_pred[:10])
    #     print (Y_true[:10])

    # accuracy is defined as frequency of events were highes prediction equal true
    best_choise = np.argmax(Y_pred, axis=1)
    print(best_choise)
    acc = (Y_true[:, best_choise]).mean()
    print("accuracy is {}".format(acc))
    return acc


# def get_wordnet_pos(word):
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def load_and_lemmatize_data(file_name):
    # a. read dataset.
    # b. lemmatizing words and save into 'headines'

    dataset = pd.read_pickle(r"C:\Users\גורים\PycharmProjects\NLP_training\datasets\News_Category_Dataset_v2_mod.pkl")
    headlines = list(dataset.headline.values)

    # lemmatizing
    lemmatizer = WordNetLemmatizer()
    print(lemmatizer.lemmatize("values"))
    new_headlines = []
    for i, sentence in enumerate(headlines):
        # if not i%2000: print (i)
        sentence = re.sub(r'\d+', ' N_digits ', sentence)
        sentence = sentence.lower()
        # new_headlines.append(" ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)]))
        new_headlines.append(" ".join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(sentence)]))
    headlines_orig = headlines
    headlines = new_headlines

    return dataset, headlines, headlines_orig

def load_processed_data(file_name):
    with open(file_name, 'rb') as f:
        d = pickle.load(f)
    return d['dataset'],d['headlines'],d['headlines_orig']