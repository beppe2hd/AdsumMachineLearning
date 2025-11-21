from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

import torch
from torch import nn
import argparse

import mysql.connector
from datetime import datetime, timedelta

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        # x: (batch_size, input_seq_len, input_size)
        outputs, hidden = self.rnn(x)  # hidden: (1, batch, hidden_size)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.rnn = nn.RNN(input_size=output_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, decoder_input, hidden):
        # decoder_input: (batch_size, 1, output_size) ← one timestep
        output, hidden = self.rnn(decoder_input, hidden)
        output = self.fc(output)  # (batch_size, 1, output_size)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output_seq_len):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_seq_len = output_seq_len

    def forward(self, x):
        batch_size = x.size(0)
        output_size = self.decoder.fc.out_features
        
        hidden = self.encoder(x)

        # Start with zeros or a special <START> token
        decoder_input = torch.zeros(batch_size, 1, output_size, device=x.device)

        outputs = []

        for _ in range(self.output_seq_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs.append(output)
            decoder_input = output  # Teacher forcing could go here

        return torch.cat(outputs, dim=1)

input_seq_len = 24 # dataset
output_seq_len = 12 # dataset
input_features_list = [0,1,2,3,4,5,6,7,8,9,10,11] # dataset
out_features_list = [0,1] # dataset
shift = 12 #dataset

#model
input_size = len(input_features_list)
output_size = len(out_features_list)
hidden_size = 20

encoder = Encoder(input_size, hidden_size)
decoder = Decoder(output_size, hidden_size)
model = Seq2Seq(encoder, decoder, output_seq_len)
model.load_state_dict(torch.load("../model_weights.pth"))
model.eval() 

class DataInput(BaseModel):
    data: List[List[float]]

def inference(model, x):
    with torch.inference_mode():
        y = model(x.unsqueeze(0))
        return  y.squeeze().detach().numpy()

app = FastAPI()

@app.post("/forecast")
def forecast(input_data: DataInput):
    # Convert to NumPy array for convenience
    x = torch.tensor(input_data.data)
    y = inference(model, x)

    return {"output": y.tolist()}