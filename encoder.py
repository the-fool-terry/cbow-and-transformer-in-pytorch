import torch
import torch.nn.init as init

def init_param():
   Wq = torch.empty(50,50,device = 'cuda' if torch.cuda.is_available() else "cpu", requires_grad=True)
   init.xavier_normal_(Wq)
   Wk = torch.empty(50,50,device = 'cuda' if torch.cuda.is_available() else "cpu", requires_grad=True)
   init.xavier_normal_(Wk)
   Wv = torch.empty(50,50,device = 'cuda' if torch.cuda.is_available() else "cpu", requires_grad=True)
   init.xavier_normal_(Wv)
   W = torch.empty(300,300,device = 'cuda' if torch.cuda.is_available() else "cpu", requires_grad=True)
   init.xavier_normal_(W)
   W1 = torch.empty(300,1200,device = 'cuda' if torch.cuda.is_available() else "cpu", requires_grad=True)
   init.xavier_normal_(W1)
   W2 = torch.empty(300,300,device = 'cuda' if torch.cuda.is_available() else "cpu", requires_grad=True)
   init.xavier_normal_(W2)
   b1 = torch.empty(1,300,device = 'cuda' if torch.cuda.is_available() else "cpu", requires_grad=True)
   init.xavier_normal_(b1)
   b2 = torch.empty(1,300,device = 'cuda' if torch.cuda.is_available() else "cpu", requires_grad=True)
   init.xavier_normal_(b2)
   gama = torch.empty(300,300,device = 'cuda' if torch.cuda.is_available() else "cpu", requires_grad=True)
   init.xavier_normal_(gama)
   bata = torch.empty(300,300,device = 'cuda' if torch.cuda.is_available() else "cpu", requires_grad=True)
   init.xavier_normal_(bata)
   return Wq,Wk,Wv,W,W1,W2,b1,b2,gama,bata



def position_encoder(e,pos,index,d = 50):
    pe = torch.zeros(e.shape)
    if index % 2 == 1:
        pe += torch.cos(pos / 10000**(2*(index // 2)/50))
    else:
        pe += torch.sin(pos / 10000**(2*(index // 2)/50))
    return pe
        
    
def self_attention(embedding,sentence,index,Wq,Wk,Wv):
    q = []
    v = []
    k = []
    for i in len(sentence):
        e = embedding[index[sentence[i]]] + position_encoder(embedding[i],i,index[sentence[i]])
        Q_word = torch.matmul(e,Wq)
        K_word = torch.matmul(e,Wk)
        V_word = torch.matmul(e,Wv)
        q.append(Q_word)
        v.append(V_word)
        k.append(K_word)
    Q = torch.stack(q)
    V = torch.stack(v)
    K = torch.stack(k)
    attention_score = torch.matmul(Q,K.T) / torch.sqrt(torch.tensor(50))
    attention_weight = torch.softmax(attention_score,dim=-1)
    output = torch.matmul(attention_weight,V)
    return output


def multi_head_attention(embedding,sentence,index,Wq,Wk,Wv,W,H=6):
    embedding_chunks = torch.chunk(embedding,dim=-1)
    outputs = []
    for e in embedding_chunks:
        output = self_attention(e,sentence,index,Wq,Wk,Wv)
        outputs.append(output)
    concatenated_output = torch.cat(outputs,dim = -1)
    final_output = torch.matmul(concatenated_output,W)
    return final_output 
    
def ffn(attention_output,W1,b1,W2,b2):
    linear_output1 = torch.matmul(attention_output, W1) + b1
    output1 = torch.relu(linear_output1)
    linear_output2 = torch.matmul(output1,W2) + b2
    output = torch.relu(linear_output2)
    return output

def res_connection(attention_output,ffn_output):
    return attention_output + ffn_output

def layer_normalized(res_output,gama, bata,eps=1e-5):
    mean = res_output.mean(dim=-1, keepdim=True)
    var = res_output.var(dim=-1,keepdim=False,unbiased=False)
    res_normalized = (res_output - mean) / torch.sqrt(var + eps)
    output = torch.matmul(res_normalized, gama) + bata
    return output


         

        

    
    



