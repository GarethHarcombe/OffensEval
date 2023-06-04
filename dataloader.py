import pandas as pd
import numpy as np

DATA_DIR = "/csse/users/grh102/Documents/cosc442/OffensEval/OLID/"

def get_dfs():    
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
    
    return df_train, df_val, df_test