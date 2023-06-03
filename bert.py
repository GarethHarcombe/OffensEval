# https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f

from transformers import BertTokenizer
from transformers import BertModel
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import pandas as pd
from eval import print_results
import itertools


DATA_DIR = "/csse/users/grh102/Documents/cosc442/cosc442-supervised-systems/OLID/"
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
        self.texts = [TOKENIZER(text, padding='max_length', max_length = 512, truncation=True,
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

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, len(LABELS))
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        
        last_hidden, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        last_hidden_masked = last_hidden * torch.reshape(mask, (1, 10, 1))
        
        # pooled embeddings
        summed_outputs = torch.sum(last_hidden_masked, dim=1)
        averaged_outputs = summed_outputs / torch.sum(mask)
        max_outputs = torch.max(last_hidden_masked[:, :torch.sum(mask), :], dim=1).values
        bert_output = pooled_output
        
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
    
    print_results(test_labels, preds)
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


def main():
    df = pd.read_csv(DATA_DIR + 'olid-training-v1.0.tsv', sep='\t')
    
    test_labels_df = pd.read_csv(DATA_DIR + 'labels-levela.csv', sep='\t', header=None, names=["id_label"])
    test_labels = [val.split(",")[1] for val in test_labels_df.id_label.values]
    
    test_docs_df = pd.read_csv(DATA_DIR + 'testset-levela.tsv', sep='\t')
    test_docs = list(test_docs_df.tweet.values)    
    
    df_test = pd.DataFrame(columns=["subtask_a", "tweet"])
    df_test["subtask_a"] = test_labels
    df_test["tweet"] = test_docs
    
    np.random.seed(112)
    df_train, df_val = np.split(df.sample(frac=1, random_state=42), 
                                [int(.9*len(df))])
    
    print(len(df_train),len(df_val), len(df_test))
                      
    EPOCHS = 4
    model = BertClassifier()
    
#    model = BertClassifier()
#    model.load_state_dict(torch.load(HOME_DIR + "model.pth"))
#    model.eval()
    
    LR = 1e-6
                  
    train(model, df_train, df_val, LR, EPOCHS)
        
    evaluate(model, df_test)
    
    torch.save(model.state_dict(), DATA_DIR + "model.pth")


#if __name__ == "__main__":
    #main()
    
model = BertClassifier()
example_text = "hello"
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_input = tokenizer(example_text,padding='max_length', max_length = 10, 
                           truncation=True, return_tensors="pt")
print(bert_input)
last_hidden, pooled_output = model(bert_input['input_ids'].squeeze(1), bert_input['attention_mask'])
print(last_hidden)