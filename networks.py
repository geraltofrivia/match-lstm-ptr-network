from __future__ import unicode_literals, print_function, division
# import matplotlib.pyplot as plt
from io import open
import numpy as np
import unicodedata
import traceback
import string
import random
import time
import re
import os


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable\

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class Encoder(nn.Module):
    
    def __init__(self, inputlen, macros, glove_file, device):
        super(Encoder, self).__init__()
        
        # Catch dim
        self.inputlen = inputlen
        self.hiddendim = macros['hidden_dim']
        self.embeddingdim =  macros['embedding_dim']
        self.vocablen = macros['vocab_size']
#         self.device = macros['device']
        self.batch_size = macros['batch_size']
        self.debug = macros['debug']
        
        # Embedding Layer
#         self.embedding = nn.Embedding(self.vocablen, self.embeddingdim)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(glove_file))
        self.embedding.weight.requires_grad = True
       
        # LSTM Layer
        self.lstm = nn.LSTM(self.embeddingdim, self.hiddendim, bidirectional=True, batch_first=False)
        
    def init_hidden(self, batch_size, device):
        
        # Returns a new hidden layer var for LSTM
        return (torch.zeros((2, batch_size, self.hiddendim), device=device), 
                torch.zeros((2, batch_size, self.hiddendim), device=device))
    
    def forward(self, x, h, device):
        
        # Input: x (batch, len ) (current input)
        # Hidden: h (1, batch, hiddendim) (last hidden state)
        
        # Batchsize: b int (inferred)
        b = x.shape[0]
        
        if self.debug > 4: print("x:\t", x.shape)
        if self.debug > 4: print("h:\t", h[0].shape, h[1].shape)
        
        x_emb = self.embedding(x)
        if self.debug > 4: 
            print("x_emb:\t", x_emb.shape)
#             print("x_emb_wrong:\t", x_emb.transpose(1,0).shape)           
            
        ycap, h = self.lstm(x_emb.transpose(1,0), h)
        if self.debug > 4: 
            print("ycap:\t", ycap.shape)
        
        return ycap, h

class MatchLSTMEncoder(nn.Module):
    
    def __init__(self, macros, device):
        
        super(MatchLSTMEncoder, self).__init__()
        
        self.hidden_dim = macros['hidden_dim']
        self.ques_len = macros['ques_len']
        self.batch_size = macros['batch_size']
        self.debug = macros['debug']    
        
        # Catch lens and params
        self.lin_g_repeat_a_dense = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.lin_g_repeat_b_dense = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.lin_g_nobias = nn.Linear(2*self.hidden_dim, self.hidden_dim, bias=False)
        
        self.alpha_i_w = nn.Parameter(torch.rand((self.hidden_dim, 1)))
        self.alpha_i_b = nn.Parameter(torch.rand((1)))
        
        self.lstm_summary = nn.LSTM((self.ques_len+1)*2*self.hidden_dim, self.hidden_dim, batch_first=False)
                                      
    
    def forward(self, H_p, h_ri, H_q, hidden, device):
        """
            Ideally, we would have manually unrolled the lstm 
            but due to memory constraints, we do it in the module.
        """
        
        # Find the batchsize
        batch_size = H_p.shape[1]
        
        H_r = torch.empty((0, batch_size, self.hidden_dim), device=device, dtype=torch.float)
        H_r = torch.cat((H_r, h_ri), dim=0)
        
        if self.debug > 4:
            print( "H_p:\t\t\t", H_p.shape)
            print( "H_q:\t\t\t", H_q.shape)
            print( "h_ri:\t\t\t", h_ri.shape)
            print( "H_r:\t\t\t", H_r.shape)
            print( "hid:\t\t\t", hidden.shape)
        
        for i in range(H_p.shape[0]):
            
            # We call the (W^P.H^P + W^rh^r_i-1 + b^P) as lin_repeat_input.
            
            # We first write out its two components as
            lin_repeat_input_a = self.lin_g_repeat_a_dense(H_p[i].view(1, batch_size, -1))
            if self.debug > 4: print("lin_repeat_input_a:\t", lin_repeat_input_a.shape)
            
            lin_repeat_input_b = self.lin_g_repeat_b_dense(H_r[i].view(1, batch_size, -1))
            if self.debug > 4: print("lin_repeat_input_b:\t", lin_repeat_input_b.shape)
            
            # Add the two terms up
            lin_repeat_input_a.add_(lin_repeat_input_b)
#             if self.debug > 4: print("lin_g_input_b unrepeated:", lin_g_input_b.shape)

            lin_g_input_b = lin_repeat_input_a.repeat(H_q.shape[0], 1, 1)
            if self.debug > 4: print("lin_g_input_b:\t\t", lin_g_input_b.shape)

            # lin_g_input_a = self.lin_g_nobias.matmul(H_q.view(-1, self.ques_len, self.hidden_dim)) #self.lin_g_nobias(H_q)
            lin_g_input_a =  self.lin_g_nobias(H_q)
            if self.debug > 4: print("lin_g_input_a:\t\t", lin_g_input_a.shape)

            G_i = F.tanh(lin_g_input_a + lin_g_input_b)
            if self.debug > 4: print("G_i:\t\t\t", G_i.shape)
            # Note; G_i should be a 1D vector over ques_len
            
            # Attention weights
            alpha_i_input_a = G_i.transpose(1,0).matmul(self.alpha_i_w).view(batch_size, 1, -1)
            if self.debug > 4: print("alpha_i_input_a:\t", alpha_i_input_a.shape)

            alpha_i_input = alpha_i_input_a.add_(self.alpha_i_b.view(-1,1,1).repeat(1,1,self.ques_len))
            if self.debug > 4: print("alpha_i_input:\t\t", alpha_i_input.shape)

            # Softmax over alpha inputs
            alpha_i = F.softmax(alpha_i_input, dim=-1)
            if self.debug > 4: print("alpha_i:\t\t", alpha_i.shape)  
                
            # Weighted summary of question with alpha    
            z_i_input_b = (
                            H_q.transpose(1, 0) *
                           (alpha_i.view(batch_size, self.ques_len, -1).repeat(1, 1, 2*self.hidden_dim))
                          ).transpose(1, 0)
            if self.debug > 4: print("z_i_input_b:\t\t", z_i_input_b.shape)

            z_i = torch.cat((H_p[i].view(1, batch_size, -1), z_i_input_b), dim=0)
            if self.debug > 4: print("z_i:\t\t\t", z_i.shape)

            # Take input from LSTM, concat in H_r and nullify the temp var.                
            h_ri, (_, hidden) = self.lstm_summary(z_i.transpose(1,0).contiguous().view(1, batch_size, -1), 
                                             (H_r[i].view(1,batch_size, -1), hidden))
            if self.debug > 4:
                print("newh_ri:\t\t", h_ri.shape)
                print("newhidden:\t\t", hidden.shape)
            H_r = torch.cat((H_r, h_ri), dim=0)
            
            if self.debug > 4:
                print("\tH_r:\t\t\t", H_r.shape)

        return H_r[1:]
    
    def init_hidden(self, batch_size, device):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return torch.zeros((1, batch_size, self.hidden_dim), device=device)
#                 torch.zeros((1, batch_size, self.hidden_dim), device=device))

class PointerDecoder(nn.Module):
    
    def __init__(self, macros, device):
        super(PointerDecoder, self).__init__()
        
        # Keep args
        self.hidden_dim = macros['hidden_dim']
        self.batch_size = macros['batch_size']
        self.para_len = macros['para_len']
        self.debug = macros['debug']
        
        self.lin_f_repeat = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin_f_nobias = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        self.beta_k_w = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.beta_k_b = nn.Parameter(torch.randn(1))
        
        self.lstm = nn.LSTM(self.hidden_dim*self.para_len, self.hidden_dim, batch_first=False)

    
    def init_hidden(self, batch_size, device):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return torch.zeros((1, batch_size, self.hidden_dim), device=device)
#                 torch.zeros((1, batch_size, self.hidden_dim), device=device))
    
    def forward(self, h_ak, H_r, hidden, device):
        
        # h_ak (current decoder's last op) (1,batch,hiddendim)
        # H_r (weighted summary of para) (P, batch, hiddendim)
        batch_size = H_r.shape[1]
        
        if self.debug > 4:
            print("h_ak:\t\t\t", h_ak.shape)
            print("H_r:\t\t\t", H_r.shape)
            print("hidden:\t\t\t", hidden.shape)
            
        # Prepare inputs for the tanh used to compute energy
        f_input_b = self.lin_f_repeat(h_ak)
        if self.debug > 4: print("f_input_b unrepeated:  ", f_input_b.shape)
        
        #H_r shape is ([PARA_LEN, BATCHSIZE, EmbeddingDIM])
        f_input_b = f_input_b.repeat(H_r.shape[0], 1, 1)
        if self.debug > 4: print("f_input_b repeated:\t", f_input_b.shape)
            
        f_input_a = self.lin_f_nobias(H_r)
        if self.debug > 4: print("f_input_a:\t\t", f_input_a.shape)
            
        # Send it off to tanh now
        F_k = F.tanh(f_input_a+f_input_b)
        if self.debug > 4: print("F_k:\t\t\t", F_k.shape) #PARA_LEN,BATCHSIZE,EmbeddingDim
        
        # Attention weights
        beta_k_input_a = F_k.transpose(1,0).matmul(self.beta_k_w).view(batch_size, 1, -1)
        if self.debug > 4: print("beta_k_input_a:\t\t", beta_k_input_a.shape)
            
        beta_k_input = beta_k_input_a.add_(self.beta_k_b.repeat(1,1,self.para_len))
        if self.debug > 4: print("beta_k_input:\t\t", beta_k_input.shape)
            
        beta_k = F.softmax(beta_k_input, dim=-1)
        if self.debug > 4: print("beta_k:\t\t\t", beta_k.shape)
        
        lstm_input_a = H_r.transpose(1,0) * (beta_k.view(batch_size, self.para_len, -1).repeat(1,1,self.hidden_dim))
        if self.debug > 4: print("lstm_input_a:\t\t", lstm_input_a.shape)
        
        h_ak, (_, hidden) = self.lstm(lstm_input_a.transpose(1,0).contiguous().view(1, batch_size, -1), (h_ak, hidden))
        
        return h_ak, hidden, F.log_softmax(beta_k_input, dim=-1)
