import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import pandas as pd
from eval import print_results
import itertools
from dataloader import get_dfs
from gensim.models import KeyedVectors
import nltk
import string
import re


DATA_DIR = "/csse/users/grh102/Documents/cosc442/OffensEval/OLID/"
MODEL_DIR = "/csse/users/grh102/Documents/cosc442/OffensEval/models/"
LABELS = {'NOT': 0, 'OFF': 1}


emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def tokenise(d):
    text_p = "".join([char for char in d.lower() if char not in string.punctuation])
    return nltk.word_tokenize(emoji_pattern.sub(r'', text_p))


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [LABELS[label] for label in df['subtask_a']]
        self.texts = [tokenise(text) for text in df['tweet']]
        self.word2vec = KeyedVectors.load("word2vec.wordvectors", mmap='r')
        self.word2vec.add_vector("[PAD]", np.zeros(self.word2vec.vector_size))

        self.texts = [[word for word in text if word in self.word2vec.index_to_key] for text in self.texts]
        self.masks = [0] * len(self.texts)
        print(self.word2vec.index_to_key)
        for i, text in enumerate(self.texts):
            # print(text)
            if len(text) > 64:
                self.texts[i] = text[:64]
            elif len(text) < 64:
                self.texts[i] = text + ["[PAD]"] * (64 - len(text))

            self.masks[i] = torch.tensor([1] * len(self.texts[i]) + [0] * (64 - len(self.texts[i])))
            self.texts[i] = torch.tensor([self.word2vec[word] for word in self.texts[i]])

        # print(self.texts)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return (self.texts[idx], self.masks[idx])

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
    
    
class Word2VecClassifier(nn.Module):

    def __init__(self, dropout=0.5, classification="regression", pooling_method="sum"):

        super(Word2VecClassifier, self).__init__()

        if classification not in ["regression", "lstm"]:
            raise ValueError("Word2VecClassifier: classification must be one of 'regression' or 'lstm'")
        self.classification = classification

        if pooling_method not in ["sum", "average", "max", "all"]:
            raise ValueError("Word2VecClassifier: pooling_method must be one of 'sum', 'average', 'max', or 'all'")
        self.pooling_method = pooling_method

        if self.classification == "regression":
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(768, len(LABELS))
            self.relu = nn.ReLU()

        if self.classification == "lstm":
            self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
            self.linear = nn.Linear(256*2, len(LABELS))

    def forward(self, text):
        print(text)
        input, mask = text
        if self.classification == "lstm":
            # https://stackoverflow.com/questions/65205582/how-can-i-add-a-bi-lstm-layer-on-top-of-bert-model
            # sequence_output has the following shape: (batch_size, sequence_length, 768)
            lstm_output, (h,c) = self.lstm(input) ## extract the 1st token's embeddings
            hidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,0, 256:]),dim=-1)
            linear_output = self.linear(hidden.view(-1,256*2)) ### assuming that you are only using the output of the last LSTM cell to perform classification
            return linear_output
        
        else:
            last_hidden_masked = text * torch.reshape(mask, (input.shape[0], input.shape[1], 1))
        
            # pooled embeddings
            if self.pooling_method == "sum":
                bert_output = torch.sum(last_hidden_masked, dim=1)
            elif self.pooling_method == "average":
                bert_output = torch.sum(last_hidden_masked, dim=1) / torch.sum(mask)
            else:
                bert_output = torch.max(last_hidden_masked[:, :torch.sum(mask), :], dim=1).values

            # regression classification
            dropout_output = self.dropout(bert_output)
            linear_output = self.linear(dropout_output)
            final_layer = self.relu(linear_output)

            return final_layer


def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    test_labels = []
    preds = []

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            
            output = model(test_input)
 
            test_labels.append(test_label)
            preds.append(output.argmax(dim=1))
            
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    
    preds       = [pred.cpu().detach().numpy()  for pred  in preds]
    preds       = list(itertools.chain.from_iterable(preds))
    test_labels = [label.cpu().detach().numpy() for label in test_labels]
    test_labels = list(itertools.chain.from_iterable(test_labels))
    
    print_results(test_labels, preds)
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


def main():
    df_train, df_val, df_test = get_dfs()

    classification = "regression"
    for pooling_method in ["sum", "average", "max"]:
        print("training BERT with {} classification method, and {} pooling".format(classification, pooling_method))
        model = Word2VecClassifier(classification=classification, pooling_method=pooling_method)
        evaluate(model, df_test)
        torch.save(model.state_dict(), MODEL_DIR + "bert_" + classification + "_" + pooling_method + ".pth")

    classification = "lstm"
    for pooling_method in ["all"]:
        print("training BERT with {} classification method, and {} pooling".format(classification, pooling_method))
        model = Word2VecClassifier(classification=classification, pooling_method=pooling_method)
        evaluate(model, df_test)
        torch.save(model.state_dict(), MODEL_DIR + "bert_" + classification + ".pth")


if __name__ == "__main__":
    main()
