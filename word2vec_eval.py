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
import gensim.downloader as api
from statistics import stdev


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


def load_sgns_embedding(filepath):
    embedding = dict()
    vocab = set()
    with open(filepath, 'r') as f:
        lines = f.readlines()
        vocab_size, embedding_size = lines[0].split()
        for i in range(1, len(lines)):
            line = lines[i].split()
            vocab.add(line[0])
            embedding[line[0]] = np.array([float(num) for num in line[1:]])

    return embedding, vocab


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [LABELS[label] for label in df['subtask_a']]
        self.texts = [tokenise(text) for text in df['tweet']]
        # self.word2vec = load_sgns_embedding("vectors.txt")
        # self.word2vec = KeyedVectors.load("word2vec.wordvectors", mmap='r')
        self.word2vec = api.load("word2vec-google-news-300")
        self.word2vec.add_vector("[PAD]", np.zeros(self.word2vec.vector_size))

        self.texts = [[word for word in text if word in self.word2vec.index_to_key] for text in self.texts]
        self.masks = [0] * len(self.texts)
        for i, text in enumerate(self.texts):
            # print(text)
            if len(text) > 64:
                self.texts[i] = text[:64]
            elif len(text) < 64:
                self.texts[i] = text + ["[PAD]"] * (64 - len(text))

            self.masks[i] = torch.unsqueeze(torch.tensor([1] * min(len(text), 64) + [0] * max((64 - len(text)), 0)), 0)
            self.texts[i] = torch.tensor([self.word2vec[word] for word in self.texts[i]])

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
            self.linear = nn.Linear(300, len(LABELS))
            self.relu = nn.ReLU()

        if self.classification == "lstm":
            self.lstm = nn.LSTM(300, 128, batch_first=True, bidirectional=True)
            self.linear = nn.Linear(128*2, len(LABELS))

    def forward(self, input, mask):
        if self.classification == "lstm":
            # https://stackoverflow.com/questions/65205582/how-can-i-add-a-bi-lstm-layer-on-top-of-bert-model
            # sequence_output has the following shape: (batch_size, sequence_length, 300)
            lstm_output, (h,c) = self.lstm(input) ## extract the 1st token's embeddings
            hidden = torch.cat((lstm_output[:,-1, :128],lstm_output[:,0, 128:]),dim=-1)
            linear_output = self.linear(hidden.view(-1,128*2)) ### assuming that you are only using the output of the last LSTM cell to perform classification
            return linear_output
        
        else:
            last_hidden_masked = input * torch.reshape(mask, (input.shape[0], input.shape[1], 1))
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


def train(model, train_data, val_data, learning_rate, epochs):

    train, val = train_data, val_data

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            
            train_label = train_label.to(device)
            input = train_input[0].to(device)
            mask = train_input[1].to(device)

            output = model(input, mask)
            
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                input = val_input[0].to(device)
                mask = val_input[1].to(device)

                output = model(input, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}')


def evaluate(model, test_data):
    test = test_data

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
            input = test_input[0].to(device)
            mask = test_input[1].to(device)

            output = model(input, mask)
 
            test_labels.append(test_label)
            preds.append(output.argmax(dim=1))
            
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    
    preds       = [pred.cpu().detach().numpy()  for pred  in preds]
    preds       = list(itertools.chain.from_iterable(preds))
    test_labels = [label.cpu().detach().numpy() for label in test_labels]
    test_labels = list(itertools.chain.from_iterable(test_labels))
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

    return print_results(test_labels, preds)


def test_many(df_train, df_val, df_test, classification, pooling_method):
    EPOCHS = 2
    LR = 1e-6
    
    print("training word2vec with {} classification method, and {} pooling".format(classification, pooling_method))

    results = []
    for i in range(5):
        model = Word2VecClassifier(classification=classification, pooling_method=pooling_method)
        train(model, df_train, df_val, LR, EPOCHS)
        results.append(evaluate(model, df_test))

    print(f"Results: {results} \n Mean: {sum(results)/len(results)} \n Stdev: {stdev(results)}")
    with open("results.txt", 'a') as f:
        f.write("training word2vec with {} classification method, and {} pooling\n".format(classification, pooling_method))
        f.write(f"Results: {results} \nMean: {sum(results)/len(results)} \nStdev: {stdev(results)}")


def main():
    df_train, df_val, df_test = get_dfs()
    df_train, df_val, df_test = Dataset(df_train), Dataset(df_val), Dataset(df_test)

    classification = "regression"
    for pooling_method in ["sum", "average", "max"]:
        # path = MODEL_DIR + "word2vec_" + classification + "_" + pooling_method + ".pth"

        test_many(df_train, df_test, df_val, classification, pooling_method)
        # torch.save(model.state_dict(), path)

    classification = "lstm"
    for pooling_method in ["all"]:
        
        # path = MODEL_DIR + "word2vec_" + classification + ".pth"
        
        test_many(df_train, df_test, df_val, classification, pooling_method)
        # torch.save(model.state_dict(), path)


if __name__ == "__main__":
    main()
