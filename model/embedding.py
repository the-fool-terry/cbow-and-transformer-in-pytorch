import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dataset.data_loader as loader
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import re

def _get_negative_samples(path,power=0.75):
   frequency = _get_frequency(path)
   freq = np.array(list(frequency.values()))
   prob = np.power(freq,power)
   prob /= np.sum(prob)
   prob = torch.tensor(prob, dtype=torch.float32)
   negative_samples = torch.multinomial(prob,15,replacement=True)
   return negative_samples



def _get_frequency(path):
   df = pd.read_json(path)
   frequency = {}
   for sentence in df[0]:
      words = re.findall(r'\b\w+\b', sentence)
      for word in words:
         if word in frequency.keys() and word.isalpha():
            frequency[word] += 1
         if word not in frequency.keys() and word.isalpha():
            frequency[word] = 1
   return frequency


def cbow(path,window_size=5,epochs = 10,learning_rate=0.0001):
   v_,u_,index,train = loader.load_data(path)
   v = torch.nn.Parameter(v_)
   u = torch.nn.Parameter(u_)
   optimizer = optim.Adam([v,u],lr=learning_rate)
   negative_samples = _get_negative_samples(path)
   for epoch in range(epochs):
     total_loss = 0
     t = 1
     for sentence in train:
        sentence_loss = 0
        for i in range(len(sentence)):
           v_average = torch.zeros(1, 300, device='cuda' if torch.cuda.is_available() else 'cpu')
           start = max(0, i - window_size)
           end = min(len(sentence), i + window_size + 1)
           count = 0 
           for j in range(start, end):
              if j == i:  
                continue
              v_average += v[index[sentence[j]]]
              count += 1
           if count > 0:
               v_average /= count
           else:
               print(f"No context words for word {i + 1}")
           positive_score = torch.matmul(u[index[sentence[i]]],v_average.T)
           negative_score = torch.matmul(u[negative_samples],v_average.T)
           loss = -torch.log(torch.sigmoid(positive_score))-torch.sum(torch.log(torch.sigmoid(-negative_score)))
           total_loss += loss.item()
           sentence_loss += loss.item()
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
        print(f'sentence_loss is {sentence_loss/(len(sentence)+1)}, in Epoch: {epoch+1} sentence: {t}')
        t += 1
     print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss}\n\n')
   
   print("train finished")
   return v,u
   

v,u = cbow("..\dataset\\train.json")
torch.save(v,"v.pth")
torch.save(u,"u.pth")   