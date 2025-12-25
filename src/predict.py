import pickle
import uvicorn
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.transform import Log1p

#model
THRESHOLD_OPTIMO = 0.1
pipeline = None

#request 
class ClientData(BaseModel):
    # Categorical attributes
    over_draft: Literal['<0', '0<=X<200', '>=200', 'no checking']
    credit_history: Literal['no credits/all paid', 'all paid', 'existing paid', 'delayed previously', 'critical/other existing credit']
    purpose: Literal['new car', 'used car', 'furniture/equipment', 'radio/tv', 'domestic appliance', 'repairs', 'education', 'vacation', 'retraining', 'business', 'other']
    Average_Credit_Balance: Literal['<100', '100<=X<500', '500<=X<1000', '>=1000', 'no known savings']
    employment: Literal['unemployed', '<1', '1<=X<4', '4<=X<7', '>=7']
    personal_status: Literal['male div/sep', 'female div/dep/mar', 'male single', 'male mar/wid', 'female single']
    property_magnitude: Literal['real estate', 'life insurance', 'car', 'no known property']
    other_payment_plans: Literal['bank', 'stores', 'none']
    housing: Literal['rent', 'own', 'for free']
    
    # Numerical attributes
    credit_usage: float = Field(ge=0.0)
    current_balance: float = Field(ge=0.0)
    location: float = Field(ge=0.0)
    cc_age: float = Field(ge=0.0)
    existing_credits: float = Field(ge=0.0)

#response 
class PredictionResult(BaseModel):
    bad_probability: float = Field(ge=0.0, le=1.0)
    fraude: bool


#app    
app=FastAPI(title="probability-fraude")


def load_model():
    global pipeline
    if pipeline is None:
        with open('./models/model_XGB.bin', 'rb') as f_in:
            pipeline = pickle.load(f_in)
    return pipeline


@app.post("/predict", response_model=PredictionResult)
def predict(client: ClientData):
    model = load_model()
    client_dict = client.model_dump()
    y_pred = model.predict_proba([client_dict])[0, 1]

    result = PredictionResult(
        bad_probability=float(f'{y_pred:.3f}'),
        fraude=bool(y_pred >= THRESHOLD_OPTIMO)
    )
    return result

    
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0",port=9696)


