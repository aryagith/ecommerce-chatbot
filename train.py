import json 
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json','r') as f:
    intents= json.load(f)

print(intents)

all_words = [] #list to hold all words
tags = [] #list to hold tags 
xy = [] #list to hold both patterns and tags

#collecting all_words from intents.json
for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern) #tokenize a pattern in list of patterns(from intents.json)
        all_words.extend(w) #use extent instead of append to make sure theres no array of arrays 
        xy.append((w, tag)) #in xy each of the tokenized words/senteces is associated with its own tag( like "greeting", "goodbye", "thanks" etc.)

ignore_words=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignore_words]#removing all the ignored charecters(?,!, etc.) and stemming the words 
all_words= sorted(set(all_words))#removing duplicates and sorting all the words.
tags=sorted(set(tags))
print(tags)

x_train = [] #bag of words list in binary (1 if the word is present in a sentence, else 0) 
y_train = [] #
for(pattern_sentence, tag) in xy:
    bag= bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label= tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples= len(x_train)
        self.x_data= x_train
        self.y_data= y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
    #Hyperparameters
    batch_size=8

    dataset=ChatDataset()
    train_loader= DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True, num_workers=2)