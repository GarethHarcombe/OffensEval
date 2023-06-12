# https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f

from transformers import BertTokenizer
from transformers import BertModel
from transformers import logging
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import pandas as pd
from eval import print_results
import itertools
from dataloader import get_dfs
from statistics import stdev


DATA_DIR = "/csse/users/grh102/Documents/cosc442/OffensEval/OLID/"
MODEL_DIR = "/csse/users/grh102/Documents/cosc442/OffensEval/models/"
LABELS = {'NOT': 0, 'OFF': 1}
TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased')

def test_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    example_text = 'I will watch Memento tonight'
    bert_input = tokenizer(example_text,padding='max_length', max_length = 10, 
                           truncation=True, return_tensors="pt")
    
    
    print(bert_input['input_ids'])
    print(bert_input['token_type_ids'])
    print(bert_input['attention_mask'])


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [LABELS[label] for label in df['subtask_a']]
        self.texts = [TOKENIZER(text, padding='max_length', max_length = 64, truncation=True,
                                return_tensors="pt") for text in df['tweet']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
    
    
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5, classification="regression", pooling_method="sum"):

        super(BertClassifier, self).__init__()

        if classification not in ["regression", "lstm"]:
            raise ValueError("BertClassifier: classification must be one of 'regression' or 'lstm'")
        self.classification = classification

        if pooling_method not in ["clt", "sum", "average", "max", "all"]:
            raise ValueError("BertClassifier: pooling_method must be one of 'clt', 'sum', 'average', 'max', or 'all'")
        self.pooling_method = pooling_method

        self.bert = BertModel.from_pretrained('bert-base-cased')

        if self.classification == "regression":
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(768, len(LABELS))
            self.relu = nn.ReLU()

        if self.classification == "lstm":
            self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
            self.linear = nn.Linear(256*2, len(LABELS))

    def forward(self, input_id, mask):
        last_hidden, pooled_output = self.bert(input_ids=input_id, attention_mask=mask,return_dict=False)
        if self.classification == "lstm":
            # https://stackoverflow.com/questions/65205582/how-can-i-add-a-bi-lstm-layer-on-top-of-bert-model
            if self.pooling_method == "clt":
                lstm_input = torch.reshape(pooled_output, (pooled_output.shape[0], 1, pooled_output.shape[1]))
            else:
                lstm_input = last_hidden

            # sequence_output has the following shape: (batch_size, sequence_length, 768)
            lstm_output, (h,c) = self.lstm(lstm_input) ## extract the 1st token's embeddings
            hidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,0, 256:]),dim=-1)
            linear_output = self.linear(hidden.view(-1,256*2)) ### assuming that you are only using the output of the last LSTM cell to perform classification
            return linear_output
        
        else:
            if self.pooling_method == "clt":
                bert_output = pooled_output
            else:
                last_hidden_masked = last_hidden * torch.reshape(mask, (last_hidden.shape[0], last_hidden.shape[1], 1))
            
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

    train, val = Dataset(train_data), Dataset(val_data)

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
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            
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
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

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
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            
            output = model(input_id, mask)
 
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
    
    results = []
    for i in range(5):
        model = BertClassifier(classification=classification, pooling_method=pooling_method)
        train(model, df_train, df_val, LR, EPOCHS)
        results.append(evaluate(model, df_test))

    print(f"Results: {results} \n Mean: {sum(results)/len(results)} \n Stdev: {stdev(results)}")

    with open("bert_results.txt", 'a') as f:
        f.write("training bert with {} classification method, and {} pooling\n".format(classification, pooling_method))
        f.write(f"Results: {results} \nMean: {sum(results)/len(results)} \nStdev: {stdev(results)}")


def main():
    logging.set_verbosity_error()

    df_train, df_val, df_test = get_dfs()

    classification = "regression"
    for pooling_method in ["sum", "clt", "average", "max"]:
        print("training BERT with {} classification method, and {} pooling".format(classification, pooling_method))
        test_many(df_train, df_val, df_test, classification, pooling_method)
        # torch.save(model.state_dict(), MODEL_DIR + "bert_" + classification + "_" + pooling_method + ".pth")

    classification = "lstm"
    for pooling_method in ["clt", "all"]:
        print("training BERT with {} classification method, and {} pooling".format(classification, pooling_method))
        test_many(df_train, df_val, df_test, classification, pooling_method)
        # torch.save(model.state_dict(), MODEL_DIR + "bert_" + classification + ".pth")


if __name__ == "__main__":
    main()
