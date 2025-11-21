from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd

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




def inference(model, x):
    with torch.inference_mode():
        y = model(x.unsqueeze(0))
        return  y.squeeze().detach().numpy()
    
def create_connection():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database='sensors'
    )

    mycursor = mydb.cursor()
    return mycursor, mydb
    
def retrive_sensor_data():

    now = datetime.now()
    date_now = pd.to_datetime(now)
    date_now_round = date_now.round("H")
    print(date_now_round)
    mycursor, mydb = create_connection()
    query = f"SELECT datetime, s_a, s_b, irr, LAI FROM sensor_data WHERE datetime = '2025-04-07 23:00:00';"
    #query = f"select * from sensor_data where datetime = '2025-04-07 23:00:00';"
    mycursor.execute(query)
    res = mycursor.fetchall()
    mycursor.close()
    mydb.close()
    return res

app = FastAPI()

@app.post("/forecast")
def forecast():
    x = [[0.7678,0.7581,-0.0427,-1.1744,-1.5878,-1.2628,1.4241,-0.2501,0.4902,-1.7477,-1.5166,-1.2903],[0.7773,0.7253,-0.0427,-1.1741,-1.5878,-1.3232,1.4241,-0.1785,0.5662,-1.7710,-1.5905,-1.2903],[0.7505,0.7433,-0.0427,-1.1739,-1.5234,-1.2628,1.4241,-0.1785,0.6137,-1.7710,-1.6274,-1.2903],[0.7735,0.7889,-0.0427,-1.1737,-1.5448,-1.1418,1.4241,-0.1785,0.6517,-1.7710,-1.7013,-1.2903],[0.7579,0.7048,-0.0427,-1.1735,-1.4160,-0.8393,1.4241,-0.2681,0.6042,-1.7477,-1.7383,-1.2903],[0.7806,0.7257,-0.0427,-1.1733,-0.7074,-1.0208,1.4241,-0.0889,0.6137,-1.5375,-1.7752,-1.2903],[0.7700,0.7557,-0.0427,-1.1731,-0.0631,-1.5047,1.4241,0.1082,0.6707,-1.1872,-1.7752,-1.2903],[0.7274,0.7231,-0.0427,-1.1728,0.3878,-1.8072,1.4241,-0.2143,0.8986,-0.8136,-1.7383,-1.2903],[0.7378,0.6985,-0.0427,-1.1726,0.5596,-1.8677,1.4241,-0.1426,0.9651,-0.5567,-1.6644,-1.2903],[0.7997,0.7289,-0.0427,-1.1724,0.7099,-1.9282,1.4241,-0.3039,1.0601,-0.3466,-1.5166,-1.2903],[0.7932,0.7505,-0.0427,-1.1722,0.9676,-2.1097,1.4241,-0.5547,1.2976,-0.0663,-1.4427,-1.2903],[0.8087,0.7326,-0.0427,-1.1720,0.9032,-1.7467,1.4241,-0.8593,1.3166,0.0271,-1.3318,-1.2903],[0.7722,0.7294,-0.0427,-1.1718,0.6240,-0.7183,1.3999,-0.8593,-1.3620,0.0037,-1.2579,-1.2903],[0.7697,0.7525,-0.0427,-1.1715,-0.0417,-0.5973,1.3272,-0.7518,-1.0106,-0.2298,-1.1840,-1.2903],[0.7689,0.7272,-0.0427,-1.1713,-0.4497,-0.2948,0.8181,-1.1639,-0.0797,-0.5801,-1.1470,-1.2903],[0.7833,0.7216,-0.0427,-1.1711,-0.9006,-0.6578,0.8666,-0.6264,0.1577,-0.8603,-1.1101,-1.2903],[0.7690,0.7520,-0.0427,-1.1709,-1.0080,-1.4442,-0.7818,0.1978,0.2432,-1.0471,-1.1470,-1.2903],[0.7273,0.7399,-0.0427,-1.1707,-0.6644,-1.8677,-0.5636,0.5740,0.2622,-1.1172,-1.1840,-1.2903],[0.7887,0.7705,-0.0427,-1.1705,-0.6644,-1.7467,1.1817,0.5023,0.2717,-1.1639,-1.1840,-1.2903],[0.7437,0.7077,-0.0427,-1.1702,-0.6859,-1.5652,-0.2727,0.4128,0.3002,-1.2106,-1.2209,-1.2903],[0.7247,0.7242,-0.0427,-1.1700,-0.4067,-1.6862,-0.5394,0.8607,0.2812,-1.2106,-1.2579,-1.2903],[0.7450,0.6685,-0.0427,-1.1698,-0.8362,-1.2628,-0.7575,0.3232,0.1862,-1.1639,-1.2209,-1.2903],[0.7362,0.7094,-0.0427,-1.1696,-0.4926,-1.5047,0.4303,0.6815,0.2622,-1.1872,-1.2579,-1.2903],[0.6908,0.7460,-0.0427,-1.1694,-0.0631,-1.7467,1.4241,0.8607,0.2717,-1.0938,-1.2948,-1.2903]]

    # Convert to NumPy array for convenience
    x = torch.tensor(x)
    y = inference(model, x)
    print(retrive_sensor_data())

    return {"output": y.tolist()}