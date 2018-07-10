"""
	coding: utf-8

	## QA over unstructured data

	Using Match LSTM, Pointer Networks, as mentioned in paper https://arxiv.org/pdf/1608.07905.pdf

	We start with the pre-processing provided by https://github.com/MurtyShikhar/Question-Answering to clean up the data and make neat para, ques files.


	### @TODOs:

	1. Figure out how to put in real, pre-trained embeddings in embeddings layer.
	2. Explicitly provide batch size when instantiating model
	3. is ./val.ids.* validation set or test set?
	4. Instead of test loss, calculate test accuracy
"""

from __future__ import unicode_literals, print_function, division
from io import open
import numpy as np
import unicodedata
import string
import random
import time
import re
import os


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable


# #### Debug Legend
# 
# - 5: Print everything that goes in every tensor.
# - 4: ??
# - 3: Check every model individually
# - 2: Print things in training loops
# - 1: ??


# Macros 
DEFAULT_DEVICE = "cuda:3"
DATA_LOC = './data/squad/'
DEBUG = 2

# nn Macros
QUES_LEN, PARA_LEN =  30, 770
VOCAB_SIZE = 115299                  # @TODO: get actual size
HIDDEN_DIM = 128
EMBEDDING_DIM = 300
BATCH_SIZE = 81                    # Might have total 100 batches.
EPOCHS = 10
TEST_EVERY_ = 5


class Encoder(nn.Module):
    
    def __init__(self, inputlen, hiddendim, embeddingdim, vocablen):
        super(Encoder, self).__init__()
        
        # Catch dim
        self.inputlen, self.hiddendim, self.embeddingdim, self.vocablen = inputlen, hiddendim, embeddingdim, vocablen
        
        # Embedding Layer
        self.embedding = nn.Embedding(self.vocablen, self.embeddingdim)
       
        # LSTM Layer
        self.lstm = nn.LSTM(self.embeddingdim, self.hiddendim)
        
    def init_hidden(self):
        
        # Returns a new hidden layer var for LSTM
        return (torch.zeros((1, BATCH_SIZE, self.hiddendim), device=device), 
                torch.zeros((1, BATCH_SIZE, self.hiddendim), device=device))
    
    def forward(self, x, h):
        
        # Input: x (1, batch, ) (current input)
        # Hidden: h (1, batch, hiddendim) (last hidden state)
        
        if DEBUG > 4: print("x:\t", x.shape)
        if DEBUG > 4: print("h:\t", h[0].shape, h[1].shape)
        
        x_emb = self.embedding(x)
        if DEBUG > 4: print("x_emb:\t", x_emb.shape)
            
        ycap, h = self.lstm(x_emb.view(-1, BATCH_SIZE, self.embeddingdim), h)
        if DEBUG > 4: print("ycap:\t", ycap.shape)
        
        return ycap, h


if DEBUG > 2:
    with torch.no_grad():

        dummy_para = torch.randint(0,VOCAB_SIZE-1,(PARA_LEN*BATCH_SIZE,), device=device).view(BATCH_SIZE,PARA_LEN).long()
    #     print (dummy_para.shape)
        dummy_question = torch.randint(0,VOCAB_SIZE-1,(QUES_LEN*BATCH_SIZE,), device=device).view(BATCH_SIZE,QUES_LEN).long()
    #     print (dummy_question.shape)

    #     print("LSTM with batches")
        ques_model = Encoder(QUES_LEN, HIDDEN_DIM, EMBEDDING_DIM, VOCAB_SIZE).cuda(device)
        para_model = Encoder(QUES_LEN, HIDDEN_DIM, EMBEDDING_DIM, VOCAB_SIZE).cuda(device)
        ques_hidden = ques_model.init_hidden()
        para_hidden = para_model.init_hidden()
        ques_embedded,hidden_ques = ques_model(dummy_question,ques_hidden)
        para_embedded,hidden_para = para_model(dummy_para,para_hidden)
        
        print (ques_embedded.shape) # question_length,batch,embedding_dim
        print (para_embedded.shape) # para_length,batch,embedding_dim
        print (hidden_para[0].shape,hidden_para[1].shape)


# ### Match LSTM
# 
# Use a match LSTM to compute a **summarized sequential vector** for the paragraph w.r.t the question.
# 
# Consider the summarized vector ($H^r$) as the output of a new decoder, where the inputs are $H^p, H^q$ computed above. 
# 
# 1. Attend the para word $i$ with the entire question ($H^q$)
#   
#     1. $\vec{G}_i = tanh(W^qH^q + repeat(W^ph^p_i + W^r\vec{h^r_{i-1} + b^p}))$
#     
#     2. *Computing it*: Here, $\vec{G}_i$ is equivalent to `energy`, computed differently.
#     
#     3. Use a linear layer to compute the content within the $repeat$ fn.
#     
#     4. Add with another linear (without bias) with $H_q$
#     
#     5. $tanh$ the bloody thing
#   
#   
# 2. Softmax over it to get $\alpha$ weights.
# 
#     1. $\vec{\alpha_i} = softmax(w^t\vec{G}_i + repeat(b))$
#     
# 3. Use the attention weight vector $\vec{\alpha_i}$ to obtain a weighted version of the question and concat it with the current token of the passage to form a vector $\vec{z_i}$
# 
# 4. Use $\vec{z_i}$ to compute the desired $h^r_i$:
# 
#     1. $ h^r_i = LSTM(\vec{z_i}, h^r_{i-1}) $
#     
# 

# In[4]:


class MatchLSTMEncoder(nn.Module):
    
    def __init__(self, hidden_dim, ques_len ):
        
        super(MatchLSTMEncoder, self).__init__()
        
        self.hidden_dim, self.ques_len = hidden_dim, ques_len
        
        # Catch lens and params
        self.lin_g_repeat = nn.Linear(2*self.hidden_dim, hidden_dim)
        self.lin_g_nobias = nn.Linear(self.hidden_dim, hidden_dim)
        
        self.alpha_i_w = nn.Parameter(torch.FloatTensor(self.hidden_dim, 1))
        self.alpha_i_b = nn.Parameter(torch.FloatTensor((1)))
        
        self.lstm_summary = nn.LSTM(self.hidden_dim*(self.ques_len+2), self.hidden_dim)
                                      
    
    def forward(self, H_p, h_ri, H_q, hidden):
        """
            Ideally, we would have manually unrolled the lstm 
            but due to memory constraints, we do it in the module.
        """
        
        H_r = torch.empty((0, BATCH_SIZE, HIDDEN_DIM), device=device, dtype=torch.float)
        H_r = torch.cat((H_r, h_ri), dim=0)
        
        if DEBUG > 4:
            print( "H_p:\t\t\t", H_p.shape)
            print( "h_ri:\t\t\t", h_ri.shape)
            print( "H_q:\t\t\t", H_q.shape)
        
        for i in range(H_p.shape[0]):
            
            lin_repeat_input = torch.cat((H_p[i].view(1, BATCH_SIZE, -1), H_r[i].view(1, BATCH_SIZE, -1)), dim=2)
            if DEBUG > 4: print("lin_repeat_input:\t", lin_repeat_input.shape)

            lin_g_input_b = self.lin_g_repeat(lin_repeat_input)
            if DEBUG > 4: print("lin_g_input_b unrepeated:", lin_g_input_b.shape)

            lin_g_input_b = lin_g_input_b.repeat(H_q.shape[0], 1, 1)
            if DEBUG > 4: print("lin_g_input_b:\t\t", lin_g_input_b.shape)

            # lin_g_input_a = self.lin_g_nobias.matmul(H_q.view(-1, self.ques_len, self.hidden_dim)) #self.lin_g_nobias(H_q)
            lin_g_input_a =  self.lin_g_nobias(H_q)
            if DEBUG > 4: print("lin_g_input_a:\t\t", lin_g_input_a.shape)

            G_i = F.tanh(lin_g_input_a + lin_g_input_b)
            if DEBUG > 4: print("G_i:\t\t\t", G_i.shape)
            # Note; G_i should be a 1D vector over ques_len

            # Attention weights
            alpha_i_input_a = G_i.view(BATCH_SIZE, -1, self.hidden_dim).matmul(self.alpha_i_w).view(BATCH_SIZE, 1, -1)
            if DEBUG > 4: print("alpha_i_input_a:\t", alpha_i_input_a.shape)

            alpha_i_input = alpha_i_input_a.add_(self.alpha_i_b.view(-1,1,1).repeat(1,1,self.ques_len))
            if DEBUG > 4: print("alpha_i_input:\t\t", alpha_i_input.shape)

            # Softmax over alpha inputs
            alpha_i = F.softmax(alpha_i_input, dim=-1)
            if DEBUG > 4: print("alpha_i:\t\t", alpha_i.shape)

            # Weighted summary of question with alpha    
            z_i_input_b = (
                            H_q.view(BATCH_SIZE, QUES_LEN, -1) *
                           (alpha_i.view(BATCH_SIZE, self.ques_len, -1).repeat(1,1,self.hidden_dim))
                          ).view(self.ques_len,BATCH_SIZE,-1)
            if DEBUG > 4: print("z_i_input_b:\t\t", z_i_input_b.shape)

            z_i = torch.cat((H_p[i].view(1, BATCH_SIZE, -1), z_i_input_b), dim=0)
            if DEBUG > 4: print("z_i:\t\t\t", z_i.shape)

            # Pass z_i, h_ri to the LSTM 
            lstm_input = torch.cat((z_i.view(1,BATCH_SIZE,-1), H_r[i].view(1, BATCH_SIZE, -1)), dim=2)
            if DEBUG > 4: print("lstm_input:\t\t", lstm_input.shape)

            # Take input from LSTM, concat in H_r and nullify the temp var.
            h_ri, hidden = self.lstm_summary(lstm_input, hidden)
            H_r = torch.cat((H_r, h_ri), dim=0)
            h_ri = None
            
            if DEBUG > 4:
                print("\tH_r:\t\t\t", H_r.shape)
#                 print("hidden new:\t\t", hidden[0].shape, hidden[1].shape)

        return H_r[1:]
    
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros((1, BATCH_SIZE, self.hidden_dim), device=device),
                torch.zeros((1, BATCH_SIZE, self.hidden_dim), device=device))

# with torch.no_grad():
#     model = MatchLSTMEncoder(HIDDEN_DIM, QUES_LEN)
#     h_pi = torch.randn(1, BATCH_SIZE, HIDDEN_DIM)
#     h_ri = torch.randn(1, BATCH_SIZE, HIDDEN_DIM)
#     hidden = model.init_hidden()
#     H_q = torch.randn(QUES_LEN, BATCH_SIZE, HIDDEN_DIM)
    
#     op, hid = model(h_pi, h_ri, H_q, hidden)
    
#     print("\nDone:op", op.shape)
#     print("Done:hid", hid[0].shape, hid[1].shape)

if DEBUG > 2:
    with torch.no_grad():
        matchLSTMEncoder = MatchLSTMEncoder(HIDDEN_DIM, QUES_LEN).cuda(device)
        hidden = matchLSTMEncoder.init_hidden()
        para_embedded = torch.rand((PARA_LEN, BATCH_SIZE, HIDDEN_DIM), device=device)
        ques_embedded = torch.rand((QUES_LEN, BATCH_SIZE, HIDDEN_DIM), device=device)
        h_ri = torch.randn(1, BATCH_SIZE, HIDDEN_DIM, device=device)
    #     if DEBUG:
    #         print ("init h_ri shape is: ", h_ri.shape)
    #         print ("the para length is ", len(para_embedded))
        H_r = matchLSTMEncoder(para_embedded.view(-1,BATCH_SIZE,HIDDEN_DIM),
                               h_ri, 
                               ques_embedded, 
                               hidden)
        print("H_r: ", H_r.shape)
        
        
        


# ### Pointer Network
# 
# Using a ptrnet over $H_r$ to unfold and get most probable spans.
# We use the **boundry model** to do that (predict start and end of seq).
# 
# A simple energy -> softmax -> decoder. Where softmaxed energy is supervised.

# In[5]:


class PointerDecoder(nn.Module):
    
    def __init__(self, hidden_dim):
        super(PointerDecoder, self).__init__()
        
        # Keep args
        self.hidden_dim = hidden_dim
        
        self.lin_f_repeat = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin_f_nobias = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        self.beta_k_w = nn.Parameter(torch.FloatTensor(self.hidden_dim, 1))
        self.beta_k_b = nn.Parameter(torch.FloatTensor(1))
        
        self.lstm = nn.LSTM(self.hidden_dim*(PARA_LEN+1), self.hidden_dim)

    
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros((1, BATCH_SIZE, self.hidden_dim), device=device),
                torch.zeros((1, BATCH_SIZE, self.hidden_dim), device=device))
    
    def forward(self, h_ak, H_r, hidden):
        
        # h_ak (current decoder's last op) (1,batch,hiddendim)
        # H_r (weighted summary of para) (P, batch, hiddendim)
        
        if DEBUG > 4:
            print("h_ak:\t\t\t", h_ak.shape)
            print("H_r:\t\t\t", H_r.shape)
            print("hidden:\t\t\t", hidden[0].shape, hidden[1].shape)
            
        # Prepare inputs for the tanh used to compute energy
        f_input_b = self.lin_f_repeat(h_ak)
        if DEBUG > 4: print("f_input_b unrepeated:  ", f_input_b.shape)
        
        #H_r shape is ([PARA_LEN, BATCHSIZE, EmbeddingDIM])
        f_input_b = f_input_b.repeat(H_r.shape[0], 1, 1)
        if DEBUG > 4: print("f_input_b repeated:\t", f_input_b.shape)
            
        f_input_a = self.lin_f_nobias(H_r)
        if DEBUG > 4: print("f_input_a:\t\t", f_input_a.shape)
            
        # Send it off to tanh now
        F_k = F.tanh(f_input_a+f_input_b)
        if DEBUG > 4: print("F_k:\t\t\t", F_k.shape) #PARA_LEN,BATCHSIZE,EmbeddingDim
            
        # Attention weights
        beta_k_input_a = F_k.view(BATCH_SIZE, -1, self.hidden_dim).matmul(self.beta_k_w).view(BATCH_SIZE, 1, -1)
        if DEBUG > 4: print("beta_k_input_a:\t\t", beta_k_input_a.shape)
            
        beta_k_input = beta_k_input_a.add_(self.beta_k_b.repeat(1,1,PARA_LEN))
        if DEBUG > 4: print("beta_k_input:\t\t", beta_k_input.shape)
            
        beta_k = F.softmax(beta_k_input, dim=-1)
        if DEBUG > 4: print("beta_k:\t\t\t", beta_k.shape)
            
        lstm_input_a = H_r.view(BATCH_SIZE, PARA_LEN, -1) * (beta_k.view(BATCH_SIZE, PARA_LEN, -1).repeat(1,1,self.hidden_dim))
        if DEBUG > 4: print("lstm_input_a:\t\t", lstm_input_a.shape)
            
        lstm_input = torch.cat((lstm_input_a.view(1, BATCH_SIZE,-1), h_ak.view(1, BATCH_SIZE, -1)), dim=2)
        if DEBUG > 4: print("lstm_input:\t\t", lstm_input.shape)
        
        h_ak, hidden = self.lstm(lstm_input, hidden)
        
        return h_ak, hidden, beta_k
            
if DEBUG > 2:
    with torch.no_grad():
        pointerDecoder = PointerDecoder(HIDDEN_DIM).cuda(device)
        h_ak = torch.randn(1,BATCH_SIZE,HIDDEN_DIM, device=device)
    #     H_r = torch.randn(PARA_LEN, BATCH_SIZE, HIDDEN_DIM)
        pointerHidden = pointerDecoder.init_hidden()
        h_ak, hidden, beta_k = pointerDecoder(h_ak, para_embedded, hidden)
        print (beta_k.shape)

# Idiot Proofing the code-so-fardef create_dummy_data(batch_size,dimension,vocab_size,max_passage_length=50,max_question_length=10):
    '''
        Create dummy data of given batch size. 
        If batch size is -1 then the function returns a pair of passage and question
    '''
    #@TODO: Implement logic for batch != -1
    min_index = 1
    max_index = vocab_size
    
    if batch_size == -1:
        passage_length = max_passage_length
        passage_node = torch.randint(min_index,max_index,(passage_length,)).long()
        question_length = max_question_length
        question_node = torch.randint(min_index,max_index,(question_length,)).long()
        answer_start_node = torch.zeros((passage_length,)).long()
        answer_start_node[passage_length-4] = 1
        answer_end_node = torch.zeros((passage_length,)).long()
        answer_end_node[passage_length-1] = 1
        return passage_node,question_node,answer_start_node,answer_end_node
    else:
        passage_length = max_passage_length
        passage_node = torch.randint(min_index,max_index,(passage_length*batch_size,)).long()
        passage_node = passage_node.view(batch_size,passage_length)
        
        question_length = max_question_length
        question_node = torch.randint(min_index,max_index,(question_length*batch_size,)).long()
        question_node = question_node.view(batch_size,question_length)
        
        answer_start_node = torch.zeros((passage_length,)).long()
        answer_start_node[passage_length-4] = 1
        answer_start_node = answer_start_node.repeat(batch_size,1).view(batch_size,-1)
        
        answer_end_node = torch.zeros((passage_length,)).long()
        answer_end_node[passage_length-1] = 1
        answer_end_node = answer_end_node.repeat(batch_size,1).view(batch_size,-1)
        
        return passage_node,question_node,answer_start_node,answer_end_node
    passage_node,question_node,answer_start_node,answer_end_node =  (create_dummy_data(12,10,10,max_passage_length=50,max_question_length=10))
print("Passage_node: ", passage_node.shape)
print("Question_node: ", question_node.shape)
print("Answer_start_node: ", answer_start_node.shape)
print("Answer_end_node: ", answer_end_node.shape)#### Testing the entire deal in a neat no_grad# with torch.no_grad():

    dummy_para = torch.randint(0,VOCAB_SIZE-1,(PARA_LEN*BATCH_SIZE,)).view(BATCH_SIZE,PARA_LEN).long()
    print (dummy_para.shape)
    dummy_question = torch.randint(0,VOCAB_SIZE-1,(QUES_LEN*BATCH_SIZE,)).view(BATCH_SIZE,QUES_LEN).long()
    print (dummy_question.shape)
    
    print("LSTM with batches")
    ques_model = Encoder(QUES_LEN, HIDDEN_DIM, EMBEDDING_DIM, VOCAB_SIZE)
    para_model = Encoder(QUES_LEN, HIDDEN_DIM, EMBEDDING_DIM, VOCAB_SIZE)
    ques_hidden = ques_model.init_hidden()
    para_hidden = para_model.init_hidden()
    ques_embedded,hidden_ques = ques_model(dummy_question,ques_hidden)
    para_embedded,hidden_para = para_model(dummy_para,para_hidden)
    
    matchLSTMEncoder = MatchLSTMEncoder(HIDDEN_DIM, QUES_LEN)
    hidden = matchLSTMEncoder.init_hidden()
    h_ri = torch.randn(1, BATCH_SIZE, HIDDEN_DIM)
    if DEBUG:
        print ("init h_ri shape is: ", h_ri.shape)
        print ("the para length is ", len(para_embedded))
    for i in range(len(para_embedded)):
        h_ri, hidden =  matchLSTMEncoder(para_embedded[i].view(1,BATCH_SIZE,-1), h_ri, ques_embedded, hidden)
        para_embedded[i] = h_ri
        DEBUG = False
    DEBUG = not DEBUG   
    
    pointerDecoder = PointerDecoder(HIDDEN_DIM)
    h_ak = torch.randn(1,BATCH_SIZE,HIDDEN_DIM)
#     H_r = torch.randn(PARA_LEN, BATCH_SIZE, HIDDEN_DIM)
    pointerHidden = pointerDecoder.init_hidden()
    h_ak, hidden, beta_k = pointerDecoder(h_ak, para_embedded, hidden)
    print (beta_k.shape)
# # Pull the real data from disk.
# 
# Files stored in `./data/squad/train.ids.*`
# Pull both train and test.

# In[6]:


train_q = np.asarray([[int(x) for x in datum.split()] for datum in list(open(os.path.join(DATA_LOC, 'train.ids.question')))])
train_p = np.asarray([[int(x) for x in datum.split()] for datum in list(open(os.path.join(DATA_LOC, 'train.ids.context')))])
train_y = np.asarray([[int(x) for x in datum.split()] for datum in list(open(os.path.join(DATA_LOC, 'train.span')))])

test_q = np.asarray([[int(x) for x in datum.split()] for datum in list(open(os.path.join(DATA_LOC, 'val.ids.question')))])
test_p = np.asarray([[int(x) for x in datum.split()] for datum in list(open(os.path.join(DATA_LOC, 'val.ids.context')))])
test_y = np.asarray([[int(x) for x in datum.split()] for datum in list(open(os.path.join(DATA_LOC, 'val.span')))])

print("Train Q: ", train_q.shape)
print("Train P: ", train_p.shape)
print("Train Y: ", train_y.shape)
print("Test Q: ", test_q.shape)
print("Test P: ", test_p.shape)
print("Test Y: ", test_y.shape)


# In[7]:


# Shuffle data
index_train, index_test = np.arange(len(train_p)), np.arange(len(test_p))
np.random.shuffle(index_train)
np.random.shuffle(index_test)

train_p, train_q, train_y = train_p[index_train], train_q[index_train], train_y[index_train]
test_p, test_q, test_y = test_p[index_test], test_q[index_test], test_y[index_test]

# Pad and prepare
train_P = np.zeros((len(train_p), PARA_LEN))
train_Q = np.zeros((len(train_q), QUES_LEN))
train_Y_start = np.zeros((len(train_p), PARA_LEN))
train_Y_end = np.zeros((len(train_p), PARA_LEN))

test_P = np.zeros((len(test_p), PARA_LEN))
test_Q = np.zeros((len(test_q), QUES_LEN))
test_Y_start = np.zeros((len(test_p), PARA_LEN))
test_Y_end = np.zeros((len(test_p), PARA_LEN))

crop_train = []    # Remove these rows from training
for i in range(len(train_p)):
    p = train_p[i]
    q = train_q[i]
    y = train_y[i]
    
    # First see if you can keep this example or not (due to size)
    if y[0] > PARA_LEN or y[1] > PARA_LEN:
        crop.append(i)
        continue
        
    
    train_P[i, :min(PARA_LEN, len(p))] = p[:min(PARA_LEN, len(p))]
    train_Q[i, :min(QUES_LEN, len(q))] = p[:min(QUES_LEN, len(q))]
    train_Y_start[i, y[0]] = 1
    train_Y_end[i, y[1]] = 1
    
crop_test = []
for i in range(len(test_p)):
    p = test_p[i]
    q = test_q[i]
    y = test_y[i]
    
    # First see if you can keep this example or not (due to size)
    if y[0] > PARA_LEN or y[1] > PARA_LEN:
        crop.append(i)
        continue
        
    test_P[i, :min(PARA_LEN, len(p))] = p[:min(PARA_LEN, len(p))]
    test_Q[i, :min(QUES_LEN, len(q))] = p[:min(QUES_LEN, len(q))]
    test_Y_start[i, y[0]] = 1
    test_Y_end[i, y[1]] = 1
    
    
# Let's free up some memory now
train_p, train_q, train_y, test_p, test_q, test_y = None, None, None, None, None, None


# # Training, and running the model
# - Write a train fn
# - Write a training loop invoking it
# - Fill in real data
# 
# ----------
# 
# Feats:
# - Function to test every n epochs.
# - Report train accuracy every epoch
# - Store the train, test accuracy for every instance.
# 

# In[8]:


def train(para_batch,
          ques_batch,
          answer_start_batch,
          answer_end_batch,
          ques_model,
          para_model,
          match_LSTM_encoder_model,
          pointer_decoder_model,
          optimizer, 
          loss_fn):
    """
	    :param para_batch: paragraphs (batch, max_seq_len_para) 
	    :param ques_batch: questions corresponding to para (batch, max_seq_len_ques)
	    :param answer_start_batch: one-hot vector denoting pos of span start (batch, max_seq_len_para)
	    :param answer_end_batch: one-hot vector denoting pos of span end (batch, max_seq_len_para)
	    
	    # Models
	    :param ques_model: model to encode ques
	    :param para_model: model to encode para
	    :param match_LSTM_encoder_model: model to match para, ques to get para summary
	    :param pointer_decoder_model: model to get a pointer over start and end span pointer
	    
	    # Loss and Optimizer.
	    :param loss_fn: 
	    :param optimizer: 
	    
	    :return: 
    """
    
    if DEBUG >=2: 
        print("\tpara_batch:\t\t", para_batch.shape)
        print("\tques_batch:\t\t", ques_batch.shape)
        print("\tanswer_start_batch:\t", answer_start_batch.shape)
        print("\tanswer_end_batch:\t\t", answer_end_batch.shape)
    
    # Wiping all gradients
    optimizer.zero_grad()
    
    # Initializing all hidden states.
    hidden_quesenc = ques_model.init_hidden()
    hidden_paraenc = para_model.init_hidden()
    hidden_mlstm = match_LSTM_encoder_model.init_hidden()
    hidden_ptrnet = pointer_decoder_model.init_hidden()
    h_ri = torch.zeros((1, BATCH_SIZE, HIDDEN_DIM), dtype=torch.float, device=device)
    h_ak = torch.zeros((1, BATCH_SIZE, HIDDEN_DIM), dtype=torch.float, device=device)
    
    if DEBUG >= 2: print("------------Instantiated hidden states------------")
    
    # Passing the data through LSTM pre-processing layer
    H_q, ques_model_hidden = ques_model(ques_batch, hidden_quesenc)
    H_p, para_model_hidden = para_model(para_batch, hidden_paraenc)
    
    if DEBUG >= 2: 
        
        print("\tH_q:\t\t", H_q.shape)
        print("\tH_p:\t\t", H_p.shape)
        print("\tH_ri:\t\t", h_ri.shape)
        raw_input("Check memory and ye shall continue")
        print("------------Encoded hidden states------------")
    
    H_r = match_LSTM_encoder_model(H_p.view(-1, BATCH_SIZE, HIDDEN_DIM), h_ri, H_q, hidden_mlstm)

    if DEBUG >= 2: print("------------Passed through matchlstm------------")
    
    
    #Passing the paragraph embddin via pointer network to generate final answer pointer.
    h_ak, hidden_ptrnet , beta_k_start = pointer_decoder_model(h_ak, H_r, hidden_ptrnet)
    h_ak, hidden_ptrnet , beta_k_end = pointer_decoder_model(h_ak, H_r, hidden_ptrnet)
    
    if DEBUG >= 2: print("------------Passed through pointernet------------")
    
    #How will we manage batches for loss.
    loss = loss_fn(beta_k_start, answer_start_batch)
    loss += loss_fn(beta_k_end, answer_end_batch)
    
    if DEBUG >= 2: print("------------Calculated loss------------")
    
    loss.backward()
    
    if DEBUG >= 2: print("------------Calculated Gradients------------")
    
    #optimization step
    optimizer.step()
    
    if DEBUG >= 2: print("------------Updated weights.------------")
    
    return loss


def training_loop():
	# Training Loop

	"""
	    > Instantiate models
	    > Instantiate loss, optimizer
	    > Instantiate ways to store loss
	    
	    > Per epoch
	        > sample batch and give to train fn
	        > get loss
	        > if epoch %k ==0: get test accuracy
	    
	    > have fn to calculate test accuracy
	"""

	DEBUG = 1

	# Instantiate models
	ques_model = Encoder(QUES_LEN, HIDDEN_DIM, EMBEDDING_DIM, VOCAB_SIZE).cuda(device)
	para_model = Encoder(PARA_LEN, HIDDEN_DIM, EMBEDDING_DIM, VOCAB_SIZE).cuda(device)
	match_LSTM_encoder_model = MatchLSTMEncoder(HIDDEN_DIM, QUES_LEN).cuda(device)
	pointer_decoder_model = PointerDecoder(HIDDEN_DIM).cuda(device)

	# Instantiate Loss
	loss_fn = nn.MSELoss()
	optimizer = optim.Adam(list(ques_model.parameters()) + 
	                       list(para_model.parameters()) + 
	                       list(match_LSTM_encoder_model.parameters()) + 
	                       list(pointer_decoder_model.parameters()))

	# Losses
	train_losses = []
	test_losses = []

	# Training Loop
	for epoch in range(EPOCHS):
	    print("Epoch: ", epoch, "/", EPOCHS)
	        
	    epoch_loss = 0.0
	    epoch_time = time.time()
	        
	    for iter in range(int(len(train_P)/BATCH_SIZE)):
	        
	        batch_time = time.time()
	        
	        # Sample batch and train on it
	        sample_index = np.random.randint(0, len(train_P), BATCH_SIZE)
	        
	        loss = train(
	            para_batch = torch.tensor(train_P[sample_index], dtype=torch.long, device=device),
	            ques_batch = torch.tensor(train_Q[sample_index], dtype=torch.long, device=device),
	            answer_start_batch = torch.tensor(train_Y_start[sample_index], dtype=torch.float, device=device).view(BATCH_SIZE, 1, PARA_LEN),
	            answer_end_batch = torch.tensor(train_Y_end[sample_index], dtype=torch.float, device=device).view(BATCH_SIZE, 1, PARA_LEN),
	            ques_model = ques_model,
	            para_model = para_model,
	            match_LSTM_encoder_model = match_LSTM_encoder_model,
	            pointer_decoder_model = pointer_decoder_model,
	            optimizer = optimizer, 
	            loss_fn= loss_fn
	        )
	    
	        epoch_loss.append(loss)

	        print("Batch:\t%d" % iter,"/%d\t: " % (len(train_P)/BATCH_SIZE),
	              "%s" % (time.time() - batch_time), 
	              "\t%s" % (time.time() - epoch_time), 
	              end=None if iter+1 == int(len(train_P)/BATCH_SIZE) else "\r")
	        
	#     print("Time taken in epoch: %s" % (time.time() - epoch_time))
	    train_losses.append(epoch_loss)
	    
	    if epoch % TEST_EVERY_ == 0:
	        pass

    return train_losses


if __name__ == "__main__":

	try:
		DEFAULT_DEVICE = sys.argv[1]
	except IndexError:
		pass

	device = torch.device(DEFAULT_DEVICE)
	torch.manual_seed(42)
	np.random.seed(42)

	# Let's start training
	training_loop()


