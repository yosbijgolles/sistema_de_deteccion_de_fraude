import pickle
import numpy as np
import pandas as pd
from scipy.io import arff
from xgboost import XGBClassifier
from dataclasses import dataclass, field
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from src.transform import Log1p
    
@dataclass
class ConfiguracionModelo:
    """Contenedor de hiperparámetros optimizados"""
    RANDOM_STATE: int = 21
    SCALE_WEIGHT: int = 5
    best_param: dict = field(default_factory=lambda:{'subsample': 0.8,
                                                    'reg_lambda': 1,
                                                     'reg_alpha': 0.1,
                                                     'n_estimators': 240,
                                                     'min_child_weight': 5,
                                                     'max_depth': 6,
                                                     'learning_rate': 0.1,
                                                     'gamma': 0.2,
                                                     'colsample_bytree': 0.9})
    

config = ConfiguracionModelo()


# CARGAR DATOS
def load_data():
    path="./data/credit_fraud.arff"
    data = arff.loadarff(path)
    df = pd.DataFrame(data[0])
    df = df.map(lambda x: x.decode() if isinstance(x, bytes) else x)
    df.rename(columns={'class': 'status'}, inplace=True)

    # features
    cat_var = ['over_draft','credit_history', 'purpose', 'Average_Credit_Balance',
                'employment', 'personal_status', 'property_magnitude',
                'other_payment_plans', 'housing','status']
    num_var = ['credit_usage','current_balance','location','cc_age','existing_credits']
    df = df[cat_var+num_var]
    return df


def train_model(df):
    y_train = (df.status=='bad').astype(int).values
    X_train = df.drop('status', axis=1).to_dict('records')

    pipeline = make_pipeline(
        Log1p(),
        DictVectorizer(),
        XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            scale_pos_weight=config.SCALE_WEIGHT,
            **config.best_param
        )
    )

    pipeline.fit(X_train, y_train)
    return pipeline

if __name__ == "__main__":
    df=load_data()

    # Preparar datos para validación cruzada
    y = (df.status=='bad').astype(int).values
    X = df.drop('status', axis=1).to_dict('records')

    # Crear pipeline para CV
    pipeline_cv = make_pipeline(
        Log1p(),
        DictVectorizer(),
        XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            scale_pos_weight=config.SCALE_WEIGHT,
            **config.best_param
        )
    )

    # Validación cruzada con AUC
    print("Calculando AUC con validación cruzada (5-fold)...")
    cv_scores = cross_val_score(pipeline_cv, X, y, cv=5, scoring='roc_auc', n_jobs=-1)

    print(f"AUC por fold: {cv_scores}")
    print(f"AUC promedio: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Entrenar modelo final con todos los datos
    print("\nEntrenando modelo final con todos los datos...")
    pipeline=train_model(df)

    # GUARDAR MODELO
    with open('./models/model_XGB.bin', 'wb') as f_out:
        pickle.dump(pipeline, f_out)

    print("El modelo se entrenó y guardó con éxito")