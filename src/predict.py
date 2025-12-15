import pickle
from pathlib import Path

project_root = Path(__file__).parent.parent
input_file = project_root / 'models' / 'model_XGB.bin'

with open(input_file, 'rb') as f_in:
    dv, model,optimal_thres = pickle.load(f_in)


def predict(client):
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    
    # threshold óptimo 
    fraude = y_pred >= optimal_thres
    
    result = {
        "bad_probability": float(round(y_pred, 2)),
        "fraude": bool(fraude)
    }
    return print(result)

