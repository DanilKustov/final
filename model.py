import re
import pandas as pd
import joblib
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split

# Загружаем данные
train = pd.read_csv("train.csv", sep=",")

ros = RandomOverSampler(random_state=42)

# Применение oversampling к признакам и меткам
X = train.drop('result', axis=1)
y = train['result']
X_resampled, y_resampled = ros.fit_resample(X, y)

# Создание нового сбалансированного датасета
train = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=['result'])], axis=1)

X = train["url"]
y = train["result"]

train = pd.concat([pd.DataFrame(X), pd.DataFrame(y, columns=['result'])], axis=1)

# Извлечем фичи из ссылок
def parse_url_features(urls):
    results = []
    for url in tqdm(urls):
        if not url:
            continue
        parsed_url = urlparse(url)
        features = {}

        features['length'] = len(url)
        features['num_subdomains'] = url.count('.')
        features['num_dots'] = url.count('.')
        features['num_slashes'] = url.count('/')
        features['path_length'] = len(parsed_url.path)
        features['is_https'] = 1 if parsed_url.scheme == 'https' else 0
        features['path_length'] = len(parsed_url.path)
        features['special_chars'] = 1 if any(char in url for char in ['#', '%', '&', '-', '_', '=', '?', '@']) else 0
        features['has_suspicious_words'] = 1 if any( word in url.lower() for word in ['login', 'secure', 'account', 'update', ]) else 0
        features['special_c_c'] = len(re.findall(r'[^a-zA-Z0-9]', url))
        features['has_numbers'] = any(char.isdigit() for char in url)
        results.append(features)
    print("pd.DataFrame(results)")
    print(pd.DataFrame(results))
    return pd.DataFrame(results)

print("x")
print(X)
features_dataframe = parse_url_features(X.to_list())
print(features_dataframe.to_numpy())
features_dataframe.head()
features_dataframe.describe()

X_train, X_test, y_train, y_test = train_test_split(
    features_dataframe.to_numpy(),
    y,
    random_state=42,
    test_size=0.2,
    shuffle=True
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train);
joblib.dump(model, "model.pkl")
