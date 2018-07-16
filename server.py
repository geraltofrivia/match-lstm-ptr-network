'''
	request json supported 
				{
				    "id": "st",
		    		"question": "str",
		    		"paragraphs": [
		      			{
		          			"id": "str",
		          			"chapter": "str",
		          			"section": "str",
		          			"paragraph": "para1"
		      			},
		      			{
		          			"id": "str",
		          			"chapter": "str",
		          			"section": "str",
		          			"paragraph": "para2"
		      			},
		      			{
		          			"id": "str",
		          			"chapter": "str",
		          			"section": "str",
		          			"paragraph": "para3"
		      			}
		    		]
				}

'''
from __future__ import unicode_literals, print_function, division
import sys
import json
import traceback
import torch
import numpy as np
from io import open
import numpy as np
import unicodedata
import traceback
import string
import random
import time
import re
import os
from utils import utils
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

from networks import Encoder, MatchLSTMEncoder, PointerDecoder
from bottle import get, request, run, response, HTTPError, post


device = torch.device("cpu")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Macros 
DATA_LOC = './data/domain/'
MODEL_LOC = './models/mlstms/domain/'
DEBUG = 1

# nn Macros
QUES_LEN, PARA_LEN =  30, 200
VOCAB_SIZE = 120000
# VOCAB_SIZE = glove_file.shape[1]               # @TODO: get actual size
HIDDEN_DIM = 150
EMBEDDING_DIM = 300
BATCH_SIZE = 100                  # Might have total 100 batches.
EPOCHS = 300
TEST_EVERY_ = 1
LR = 0.001
CROP = None

macros = {
    "ques_len": QUES_LEN,
    "hidden_dim": HIDDEN_DIM, 
    "vocab_size": VOCAB_SIZE, 
    "batch_size": BATCH_SIZE,
    "para_len": PARA_LEN,
    "embedding_dim": EMBEDDING_DIM,
    "lr": LR,
    "debug":DEBUG,
    "save_model_loc": MODEL_LOC,
    'data_loc' : DATA_LOC
}

vectors = np.load(os.path.join(macros['data_loc'], 'glove.new.trimmed.300.npy'))

ques_model = torch.load(os.path.join(macros['save_model_loc'], 'ques_model.torch')
	,map_location=lambda storage, location: storage)
print("Ques Model\n", ques_model)

para_model = torch.load(os.path.join(macros['save_model_loc'], 'para_model.torch')
	,map_location=lambda storage, location: storage)
print("Para Model\n", para_model)

mlstm_model = torch.load(os.path.join(macros['save_model_loc'], 'mlstm_model.torch')
	,map_location=lambda storage, location: storage)
print("MLSTM Model\n", mlstm_model)

pointer_decoder_model = torch.load(os.path.join(macros['save_model_loc'], 'pointer_decoder_model.torch')
	,map_location=lambda storage, location: storage)
print("Pointer Decoder model\n", pointer_decoder_model)


URL = 'localhost'
PORT = 9000
GPU = 3
VOCAB_FILE = 'resources/vocab.dat'
word_to_id = {}
id_to_word = {}
#@TODO: autmatically identify this.
unknown = '<unk>'

def predict(para_batch,
            ques_batch,
            ques_model,
            para_model,
            mlstm_model,
            pointer_decoder_model,
            macros,
            loss_fn=None,
            debug=DEBUG):
    """
        Function which returns the model's output based on a given set of P&Q's. 
        Does not convert to strings, gives the direct model output.
        
        Expects:
            four models
            data
            misc macros
    """
    
#     BATCH_SIZE = macros['batch_size']
    BATCH_SIZE = ques_batch.shape[0]
    HIDDEN_DIM = macros['hidden_dim']
    DEBUG = debug
    
    if debug >=2: 
        print("\tpara_batch:\t\t", para_batch.shape)
        print("\tques_batch:\t\t", ques_batch.shape)
        
    with torch.no_grad():    

        # Initializing all hidden states.
        hidden_quesenc = ques_model.init_hidden(BATCH_SIZE, device)
        hidden_paraenc = para_model.init_hidden(BATCH_SIZE, device)
        hidden_mlstm = mlstm_model.init_hidden(BATCH_SIZE, device)
        hidden_ptrnet = pointer_decoder_model.init_hidden(BATCH_SIZE, device)
        h_ri = torch.zeros((1, BATCH_SIZE, HIDDEN_DIM), dtype=torch.float, device=device)
        h_ak = torch.zeros((1, BATCH_SIZE, HIDDEN_DIM), dtype=torch.float, device=device)
        if DEBUG >= 2: print("------------Instantiated hidden states------------")
            
        #passing the data through LSTM pre-processing layer
        H_q, ques_model_hidden = ques_model(ques_batch, hidden_quesenc, device)
        H_p, para_model_hidden = para_model(para_batch, hidden_paraenc, device)
        if DEBUG >= 2: 
            print("\tH_q:\t\t", H_q.shape)
            print("\tH_p:\t\t", H_p.shape)
            print("\tH_ri:\t\t", h_ri.shape)
#             raw_input("Check memory and ye shall continue")
            print("------------Encoded hidden states------------")

        H_r = mlstm_model(H_p.view(-1, BATCH_SIZE, 2*HIDDEN_DIM), h_ri, H_q, hidden_mlstm, device)
        if DEBUG >= 2: print("------------Passed through matchlstm------------")

        #Passing the paragraph embddin via pointer network to generate final answer pointer.
        h_ak, hidden_ptrnet, beta_k_start = pointer_decoder_model(h_ak, H_r, hidden_ptrnet, device)
        _, _, beta_k_end = pointer_decoder_model(h_ak, H_r, hidden_ptrnet, device)
        if DEBUG >= 2: print("------------Passed through pointernet------------")
                            
        # For crossentropy
#         _, answer_start_batch = answer_start_batch.max(dim=2)[1]
#         _, answer_end_batch = answer_end_batch.max(dim=2)[1]
#         print("labels: ", answer_start_batch.shape)[1]
            
#         #How will we manage batches for loss.
#         loss = loss_fn(beta_k_start, answer_start_batch)
#         loss += loss_fn(beta_k_end, answer_end_batch)
#         if debug >= 2: print("------------Calculated loss------------")
            
        return (beta_k_start, beta_k_end, 0.0)

def load():
	global word_to_id,id_to_word

	print("loading vocab file")
	vocab = utils.load_file(VOCAB_FILE)
	for index,v in enumerate(vocab):
		word_to_id[v] = index
		id_to_word[index] = v

def text_to_id(text):
	'''
		Text - input which will be converted to string of ids
	'''
	processed_text = utils.preProcessing(text)
	for index in range(len(processed_text)):
		try:
			processed_text[index] = word_to_id[processed_text[index]]
		except KeyError:
			processed_text[index] = word_to_id[unknown]
		except:
			print(traceback.print_exc())
			raise ValueError

	return np.array(processed_text)

def predict_span(question_id,paragraph_id):
	'''
		Returns the span of the paragraph_id. 
	'''
	padded_question_id = np.zeros((1,macros['ques_len']))
	padded_para_id = np.zeros((1,macros['para_len']))
	padded_question_id[0,:min(len(question_id),macros['ques_len'])] = question_id[:min(len(question_id),macros['ques_len'])]
	padded_para_id[0,:min(len(paragraph_id),macros['para_len'])] = paragraph_id[:min(len(paragraph_id),macros['para_len'])].reshape(1,-1)
	y_cap_start, y_cap_end, _ = predict(torch.tensor(padded_para_id, dtype=torch.long, device=device), 
                                   torch.tensor(padded_question_id, dtype=torch.long, device=device),
                                   ques_model=ques_model,
                                   para_model=para_model,
                                   mlstm_model=mlstm_model,
                                   pointer_decoder_model=pointer_decoder_model,
                                    macros=macros,
                                    debug=macros['debug'])
	y_cap_start,y_cap_end = torch.argmax(y_cap_start.squeeze(1), dim=1).float(), torch.argmax(y_cap_end.squeeze(1), dim=1).float()
	return y_cap_start.item(),y_cap_end.item()


def retrive_span(start_index,end_index,paragraph):
	para = utils.preProcessing(paragraph)
	if end_index > start_index :
		return " ".join(para[int(start_index):int(end_index)+1])
	else:
		return para[int(start_index)]


def start(URL,PORT):
	load()
	run(host=URL, port=PORT)

@post('/answer')
def answer():
	try:
		try:
			print(request.json)
			data = request.json
		except:
			raise ValueError

		if data is None:
			raise ValueError

		try:
			question = data['question']
			question_id = text_to_id(question)
			print(question_id)
			paragraphs = data['paragraphs']
			for index,para in enumerate(paragraphs):
				paragraph_id = text_to_id(para['paragraph'])
				start_index,end_index = predict_span(question_id, paragraph_id)
				v = retrive_span(int(start_index), int(end_index), para['paragraph'])
				paragraphs[index]['span'] = v
				paragraphs[index]['score'] = 1
				print (paragraphs[index]['span'])
			data['paragraphs'] = paragraphs
		except (ValueError,KeyError):
			print(traceback.print_exc())
			raise ValueError
		return data
	except ValueError:
		print(traceback.print_exc())
		response.status = 400
		return
	except KeyError:
		print(traceback.print_exc())
		response.status = 409
		return

if __name__== "__main__":
	try:
		URL = sys.argv[1]
	except IndexError:
		pass
	try:
		PORT = int(sys.argv[2])
	except IndexError,TypeError:
		pass

	print("About to start server on %(url)s:%(port)s" % {'url':URL, 'port':str(PORT)})
	start(URL,PORT)
