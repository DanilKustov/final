from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from urllib.parse import urlparse
import pandas as pd
import re

app = FastAPI()

# Загружаем предобученную модель
model = joblib.load("model.pkl")


# Определяем входные данные, которые будем принимать через API
class URLData(BaseModel):
    url: str

# Создаем маршрут для предсказания
@app.post("/predict/")
def predict(data: URLData):
    try:
        # Здесь выполняем обработку данных и извлечение признаков из data.url
        features = parse_url_features(data.url)  # Ваша функция для извлечения признаков
        features= features.to_numpy()
        # Делаем предсказание
        prediction = model.predict(features)
        print(prediction)
        result = "Phishing" if prediction[0] == 1 else "Safe"

        return {"url": data.url, "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def parse_url_features(urls):
    results = []

    parsed_url = urlparse(urls)
    features = {}

    features['length'] = len(urls)
    features['num_subdomains'] = urls.count('.')
    features['num_dots'] = urls.count('.')
    features['num_slashes'] = urls.count('/')
    features['path_length'] = len(parsed_url.path)
    features['is_https'] = 1 if parsed_url.scheme == 'https' else 0
    features['path_length'] = len(parsed_url.path)
    features['special_chars'] = 1 if any(char in urls for char in ['#', '%', '&', '-', '_', '=', '?', '@']) else 0
    features['has_suspicious_words'] = 1 if any( word in urls.lower() for word in ['login', 'secure', 'account', 'update', ]) else 0
    features['special_c_c'] = len(re.findall(r'[^a-zA-Z0-9]', urls))
    features['has_numbers'] = any(char.isdigit() for char in urls)
    results.append(features)
    print("pd.DataFrame(results)")
    print(pd.DataFrame(results))
    return pd.DataFrame(results)