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
from bottle import get, request, run, response, HTTPError, post
from io import open
# import matplotlib.pyplot as plt
import collections
import numpy as np
import unicodedata
import traceback
import random
import string
import torch
import json
import time
import sys
import re
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import utils
from networks import Encoder, MatchLSTMEncoder, PointerDecoder

device = torch.device("cpu")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Macros 
DATA_LOC = './data/domain/'
MODEL_LOC = './models/mlstms/domain/'
VOCAB_FILE = './models/mlstms/domain/vocab.dat'
DEBUG = 1

# Neural Network Macros (some useless)
QUES_LEN, PARA_LEN =  30, 200
VOCAB_SIZE = 120000
HIDDEN_DIM = 150
EMBEDDING_DIM = 300
BATCH_SIZE = 50                  # Might have total 100 batches.
EPOCHS = 300
TEST_EVERY_ = 1
LR = 0.001
CROP = None

# Server macros
URL = 'localhost'
PORT = 9000
GPU = 3
word_to_id = {}
id_to_word = {}
#@TODO: autmatically identify this.
unknown = '<unk>'

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

# Open glove vectors
vectors = np.load(os.path.join(macros['data_loc'], 'glove.new.trimmed.300.npy'))

# Load models
ques_model = torch.load(os.path.join(macros['save_model_loc'], 'ques_model.torch')
	,map_location=lambda storage, location: storage)
if macros['debug'] > 3: print("Ques Model\n", ques_model)

para_model = torch.load(os.path.join(macros['save_model_loc'], 'para_model.torch')
	,map_location=lambda storage, location: storage)
if macros['debug'] > 3: print("Para Model\n", para_model)

mlstm_model = torch.load(os.path.join(macros['save_model_loc'], 'mlstm_model.torch')
	,map_location=lambda storage, location: storage)
if macros['debug'] > 3: print("MLSTM Model\n", mlstm_model)

pointer_decoder_model = torch.load(os.path.join(macros['save_model_loc'], 'pointer_decoder_model.torch')
	,map_location=lambda storage, location: storage)
if macros['debug'] > 3: print("Pointer Decoder model\n", pointer_decoder_model)



def predict(para_batch, ques_batch, ques_model, para_model, mlstm_model, pointer_decoder_model, macros, loss_fn=None, debug=DEBUG):
    """
        Function which returns the model's output based on a given set of P&Q's. 
        Does not convert to strings, gives the direct model output.
        
        Expects:
            four models
            data
            misc macros

        *note: does not return the loss (does not bother)*
    """
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
            
        # Passing the data through LSTM pre-processing layer
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
            
        return (beta_k_start, beta_k_end, 0.0)


def prepare_words():
	"""
		Function to be called during server init.
		Loads the word to ID and ID to word dictionaries, in keeping with model vocabulary.
	"""
	global word_to_id,id_to_word

	if DEBUG >=2: print("Loading vocabulary from disk")
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


def predict_span(question_id,paragraph_id,batch_size = None):
	'''
		Returns the span of the paragraph_id. 
	'''
	padded_question_id = np.zeros((1,macros['ques_len']))
	padded_para_id = np.zeros((1,macros['para_len']))
	padded_question_id[0,:min(len(question_id),macros['ques_len'])] = question_id[:min(len(question_id),macros['ques_len'])]
	padded_para_id[0,:min(len(paragraph_id),macros['para_len'])] = paragraph_id[:min(len(paragraph_id),macros['para_len'])].reshape(1,-1)
	
	if batch_size:
		padded_para_id = np.repeat(padded_para_id,batch_size,axis=0)
		padded_question_id = np.repeat(padded_question_id,batch_size,axis=0)


	# Pass the padded versions to the predict function, get model outputs.
	y_cap_start, y_cap_end, _ = predict(torch.tensor(padded_para_id, dtype=torch.long, device=device), 
                                   torch.tensor(padded_question_id, dtype=torch.long, device=device),
                                   ques_model=ques_model,
                                   para_model=para_model,
                                   mlstm_model=mlstm_model,
                                   pointer_decoder_model=pointer_decoder_model,
                                    macros=macros,
                                    debug=macros['debug'])

	# Find the max ID from the model outputs to get start and end span.
	if batch_size:
		y_cap_start,y_cap_end = torch.argmax(y_cap_start.squeeze(), dim=0),torch.argmax(y_cap_end.squeeze(), dim=0)
		print("most common are ,", collections.Counter(y_cap_start).most_common()[0][0].item(),collections.Counter(y_cap_end).most_common()[0][0].item())
		return collections.Counter(y_cap_start).most_common()[0][0],collections.Counter(y_cap_end).most_common()[0][0]
	else:
		y_cap_start,y_cap_end = torch.argmax(y_cap_start.squeeze(1), dim=1).float(), torch.argmax(y_cap_end.squeeze(1), dim=1).float()	
		return y_cap_start.item(),y_cap_end.item()


def retrive_span(start_index,end_index,paragraph):
	'''
		Given span, return the chunk of text retrieved
	'''
	para = utils.preProcessing(paragraph)
	if end_index > start_index :
		return " ".join(para[int(start_index):int(end_index)+1])
	else:
		return para[int(start_index)]

# Start the server
def start(URL,PORT):
	prepare_words()
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
				start_index,end_index = predict_span(question_id, paragraph_id,None)
				paragraphs[index]['span'] = retrive_span(int(start_index), int(end_index), para['paragraph'])
				paragraphs[index]['score'] = 1
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
