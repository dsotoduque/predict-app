import fastapi
import json
from model import DelayModel
import pandas as pd

app = fastapi.FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict() -> dict:
    data = pd.read_csv('./data/data.csv')
    delay_model = DelayModel()
    features = delay_model.preprocess(data)
    delay_model.fit(features=features, target=features['delay'])
    predict_result = delay_model.predict()
    return json.dumps(predict_result)