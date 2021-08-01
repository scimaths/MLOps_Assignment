import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

import pycaret.classification as pycl

from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()
app = FastAPI()

class Model:
    def __init__ (self, modelname, bucketname):
        self.model = pycl.load_model(modelname, platform = 'aws', authentication = { 'bucket' : bucketname })
    
    def predict (self, data):
        predictions = pycl.predict_model(self.model, data = data).Label.to_list()
        return predictions


lgbm = Model("lgbm_deployed", "mlopsfinalassignment200050148")
dt = Model("dt_deployed", "mlopsfinalassignment200050148")

@app.post("/lgbm/predict")
async def create_upload_file(file:UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)
        try:
            k = lgbm.predict(data)
        except:
            raise HTTPException(status_code=404, detail="Invalid CSV file")
        return {
            "Labels": k
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")

@app.post("/dt/predict")
async def create_upload_file(file:UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)
        try:
            k = dt.predict(data)
        except:
            raise HTTPException(status_code=404, detail="Invalid CSV file")
        return {
            "Labels": k
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")

if os.getenv("AWS_ACCESS_KEY_ID") == None or os.getenv("AWS_SECRET_ACCESS_KEY") == None:
    print("AWS Credentials missing. Please set required environment variables.")
    exit(1)