import pickle
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

# CARGAR DATOS
data = arff.loadarff("../data/credit_fraud.arff")
df = pd.DataFrame(data[0])
df = df.map(lambda x: x.decode() if isinstance(x, bytes) else x)
df["status"] = df["class"]
del df["class"]
df['current_balance'] = np.log1p(df['current_balance'])

# FEATURES SELECCIONADOS
cat = ['over_draft','credit_history', 'purpose', 'Average_Credit_Balance', 
       'employment', 'personal_status', 'property_magnitude', 
       'other_payment_plans', 'housing', 'status']
num = ['credit_usage','current_balance','location','cc_age','existing_credits']
df = df[cat+num]

# SPLIT
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=11, stratify=df['status'])
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=11, stratify=df_train_full['status'])

# TARGET
y_train = (df_train.status=='bad').astype(int).values
y_val = (df_val.status=='bad').astype(int).values
y_train_full = (df_train_full.status=='bad').astype(int).values
y_test = (df_test.status=='bad').astype(int).values

# FEATURES
X_train = df_train.drop('status', axis=1)
X_val = df_val.drop('status', axis=1)
X_train_full = df_train_full.drop('status', axis=1)
X_test = df_test.drop('status', axis=1)

# VECTORIZAR
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(X_train.to_dict(orient='records'))
X_val = dv.transform(X_val.to_dict(orient='records'))

dv_full = DictVectorizer(sparse=False)
X_train_full = dv_full.fit_transform(X_train_full.to_dict(orient='records'))
X_test = dv_full.transform(X_test.to_dict(orient='records'))

# HYPERPARAMETER SEARCH
param_distributions = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1, 2],
    'n_estimators': range(150, 350, 10)
}

model = XGBClassifier(objective='binary:logistic', eval_metric='auc', 
                      random_state=42, n_jobs=-1, scale_pos_weight=5)

search = RandomizedSearchCV(model, param_distributions, n_iter=30, 
                           scoring='roc_auc', cv=5, random_state=21, 
                           n_jobs=-1, verbose=1)

search.fit(X_train, y_train)

# REENTRENAR CON TRAIN_FULL
model_final = XGBClassifier(objective='binary:logistic', eval_metric='auc', 
                            random_state=42, n_jobs=-1, scale_pos_weight=5, 
                            **search.best_params_)
model_final.fit(X_train_full, y_train_full)






# SETEAR EL THRESHOLD OPTIMO
y_pred_proba = model_final.predict_proba(X_test)[:, 1]  # Probabilidad de STATUS=1 (CLASS=BAD)

# RANGO DE BUSQUEDA
thresholds = np.arange(0.1, 0.95, 0.05)
thres_cost = []

for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
    
    # FN = DEJAMOS PASAR FRAUDE   vs FP = BLOQUEMOS UN NO FRAUDE
    cost = 5*fn  + 1*fp
    thres_cost.append({'threshold': threshold,'cost': cost})

optimal = min(thres_cost, key=lambda x: x['cost'])
optimal_thres=optimal["threshold"]





# GUARDAR MODELO
with open('../models/model_XGB.bin', 'wb') as f:
    pickle.dump((dv_full, model_final,optimal_thres), f)

print("El modelo de se entrenó y guardó con éxito")