import random
import json
import torch
from model import NeuralNetwork
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pt"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNetwork(input_size=len(all_words), hidden_size=8, output_size=len(tags)).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name="David"

print("Let's chat! Type 'quit' to exit.")

while True:
    sentence = input("User:")
    if sentence == "quit":
        break

    sentence  = tokenize(sentence)
    x = bag_of_words(sentence,all_words)
    x = x.reshape(1,x.shape[0])
    x = torch.from_numpy(x)

    output= model(x)
    _, predicted = torch.max(output, dim=1)
    tag= tags[predicted.item()]

    probs=torch.softmax(output, dim=1) # applying softmax to get the probability of output 
    prob= probs[0][predicted.item()] #prob is the probability of the predicted output by the network
    
    if prob.item() > 0.75: #only respond with answer if the probability of a certain tag is high enough.
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent["responses"])}")
    else:
        print(f"{bot_name}: I do not understand...")