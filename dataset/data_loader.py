import pandas as pd
import torch.nn.init as init
import torch
import re

def load_data(path):
    v = {}
    train = []
    df = pd.read_json(path)
    for sentence in df[0]:
        train.append([])
        words = re.findall(r'\b\w+\b', sentence)
        for word in words:
            if word.isalpha():
                train[-1].append(word) 
                if word not in v.keys():
                  embedding = torch.empty(1,300, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True)
                  init.xavier_normal_(embedding)
                  v[word] = embedding
    keys = v.keys()
    alpha_index = {key:index for index,key in enumerate(keys)}
    v_ = torch.stack(list(v.values()), dim=0)
    u = torch.empty(v_.shape,device="cuda" if torch.cuda.is_available() else 'cpu',requires_grad=True)
    init.xavier_normal_(u)
    return v_,u,alpha_index,train
    


