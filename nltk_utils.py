import nltk
from nltk.stem.porter import PorterStemmer #importing stemmer from nltk library
import numpy as np
#nltk.download('punkt') #downloading pretrained tokenizer from nltk library

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence) #tokenising the sentence into words/tokens

def stem(word):
    return stemmer.stem(word.lower())#making the tokens/words lowercase and stemming them

def bag_of_words(tokenized_sentence, all_words):
    """
    example for bag of words :-
    sentence = ["hello", "how", "are", "you"] 
    all_words = ["hi","hello", "I","you","bye","thank", "cool"]
    bag =       [0,    1,     0,     1,     0,    0,      0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag= np.zeros(len(all_words),dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag



