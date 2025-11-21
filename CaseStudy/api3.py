from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI()

# Define the request body model
class DataInput(BaseModel):
    data: List[List[float]]

@app.post("/mean")
def compute_mean(input_data: DataInput):
    # Convert to NumPy array for convenience
    arr = np.array(input_data.data)
    
    # Compute overall mean
    mean_value = float(arr.mean())

    return {"mean": mean_value}