# Sistema de Detección de Fraude

Éste sistema encapsula un clasificador de fraude usando XGBoost.

## Estructura

```
├── data/
│   └── credit_fraud.arff          # Dataset de entrenamiento
├── models/
│   └── model_XGB.bin              # Modelo entrenado
├── notebooks/
│   ├── EDA.ipynb                  # Análisis exploratorio
│   └── XGB.ipynb                  # Entrenamiento XGBoost
├── src/
│   ├── train.py                   # Entrena el modelo
│   └── predict.py                 # Predicciones
└── test.py                        # Ejemplo de uso
```

## Instalación

```bash
pipenv install
pipenv shell
```

## Uso

### 1. Entrenar modelo: 
Después de un analisis EDA exhaustivo se seleccionaron un grupo de features 
con mayor poder predictivo para el el clasificador. Bajo estas caracteristicas 
se optimizo un modelo XGBoost para su posterior entrenamiento.

```bash
python src/train.py
```

Genera `models/model_XGB.bin` con:
- Vectorizador de características
- Modelo XGBoost optimizado
- Threshold óptimo (balance costo 5:1 para FN/FP)

### 2. Predecir

```python
from src.predict import predict

client = {
    'over_draft': 'no checking',
    'credit_history': 'critical/other existing credit',
    'purpose': 'radio/tv',
    'Average_Credit_Balance': '1 - 200',
    'employment': '4 - 7 years',
    'personal_status': 'female div/dep/mar',
    'property_magnitude': 'real estate',
    'other_payment_plans': 'none',
    'housing': 'own',
    'credit_usage': 2.0,
    'current_balance': 1169.0,
    'location': 67.0,
    'cc_age': 4.0,
    'existing_credits': 2.0
}

resultado = predict(client)
# {'bad_probability': 0.123, 'fraude': False}
```

## Características

Tras el EDA se selecccionaron las siguientes carácteristicas:

- **10 categóricas**: over_draft, credit_history, purpose, Average_Credit_Balance, employment, personal_status, property_magnitude, other_payment_plans, housing
- **5 numéricas**: credit_usage, current_balance, location, cc_age, existing_credits

## Modelo

- **Algoritmo**: XGBoost Classifier
- **Optimización**: RandomizedSearchCV (30 iteraciones, 5-fold CV)
- **Balance de clases**: scale_pos_weight=5
- **Threshold**: Calibrado por matriz de costos
