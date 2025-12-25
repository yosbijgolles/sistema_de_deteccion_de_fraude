# Sistema de Detección de Fraude Crediticio

API REST de predicción de riesgo crediticio usando XGBoost. Predice probabilidad de "mal crédito" basándose en 14 características financieras y demográficas.

**Performance:** AUC = 0.77 (5-fold CV) | **Threshold:** 0.10 | **Modelo:** 422KB

## Instalación

```bash
# Con uv (recomendado)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Con pip
pip install fastapi numpy pandas scikit-learn scipy uvicorn xgboost
```

<<<<<<< HEAD
## Uso

### 1. Entrenar modelo: 
Después de un analisis EDA exhaustivo se seleccionaron un grupo de features 
con mayor poder predictivo para el el clasificador. Bajo estas caracteristicas 
se optimizo un modelo XGBoost para su posterior entrenamiento.
=======
## Uso Rápido
>>>>>>> 6a2ffb6 (xgboost API REST)

```bash
# 1. Entrenar modelo
uv run python src/train.py

# 2. Iniciar API
uv run uvicorn src.predict:app --host 0.0.0.0 --port 9696

# 3. Probar
uv run python test.py
```

## API

**Endpoint:** `POST /predict`

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "over_draft": "no checking",
    "credit_history": "critical/other existing credit",
    "purpose": "education",
    "Average_Credit_Balance": "no known savings",
    "employment": "1<=X<4",
    "personal_status": "female div/dep/mar",
    "property_magnitude": "car",
    "other_payment_plans": "none",
    "housing": "own",
    "credit_usage": 24.0,
    "current_balance": 1926.0,
    "location": 3.0,
    "cc_age": 33.0,
    "existing_credits": 2.0
  }'
```

**Respuesta:**
```json
{"bad_probability": 0.091, "fraude": false}
```

<<<<<<< HEAD
Tras el EDA se selecccionaron las siguientes carácteristicas:
=======
## Docker
>>>>>>> 6a2ffb6 (xgboost API REST)

```bash
docker build -t fraud-detection .
docker run -p 9696:9696 fraud-detection
```

## Deployment (Fly.io)

```bash
fly launch    # primera vez
fly deploy    # actualizar
```

**Producción:** https://sistema-de-deteccion-de-fraude.fly.dev

## Estructura

```
├── data/credit_fraud.arff      # Dataset
├── models/model_XGB.bin        # Pipeline entrenado
├── notebooks/                  # EDA y experimentación
├── src/
│   ├── train.py               # Entrenamiento
│   ├── predict.py             # API FastAPI
│   └── transform.py           # Transformer Log1p
├── Dockerfile
├── fly.toml
└── test.py
```

## Modelo

- **Algoritmo:** XGBoost (240 estimators, max_depth=6)
- **Features:** 9 categóricas + 5 numéricas
- **Pipeline:** Log1p → DictVectorizer → XGBClassifier
- **Balance:** scale_pos_weight=5
- **Optimización:** RandomizedSearchCV (30 iter, 5-fold CV)

## Features Principales

**Categóricas:** over_draft, credit_history, purpose, Average_Credit_Balance, employment, personal_status, property_magnitude, other_payment_plans, housing

**Numéricas:** credit_usage, current_balance (log1p), location, cc_age, existing_credits
