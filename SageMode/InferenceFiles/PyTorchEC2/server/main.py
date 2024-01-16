from fastapi import FastAPI
import torch
from model import model
from pre_process import pre_process
from post_process import post_process

path = "weights.pth"

model.load_state_dict(torch.load(path))
model.eval()
app = FastAPI()

@app.post("/predict")
def predict(data:dict) -> dict:
    kwargs = pre_process(data)
    raw_output = model(**kwargs)
    post_process_result = post_process(raw_output)
    return post_process_result
