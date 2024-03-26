# --- Imports ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import time

# --- Preprocessor Definition ---
numeric_features = ['visibility', 'dew_point', 'feels_like', 'temp_min', 'temp_max',
                    'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all',
                    'rain_1h', 'rain_3h']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_features = ['weather_main', 'weather_description', 'weather_icon']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# --- Functions ---
def assign_modeling_roles(launch_data):
    """Preprocesses the launch data by handling missing values and encoding categorical variables."""

    target = ['LAUNCHED']
    predictors = ['visibility', 'dew_point', 'feels_like', 'temp_min', 'temp_max',
                  'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all',
                  'rain_1h', 'rain_3h', 'weather_main', 'weather_description',
                  'weather_icon']

    X = launch_data[predictors]
    y = launch_data[target]

    # Splitting the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Evaluates the performance of multiple models on the given data."""

    results = []

    for name, model in models.items():
        # Timing model training and prediction
        start_time = time.time()

        # Create and fit the model pipeline
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('classifier', model)])
        model_pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = model_pipeline.predict(X_test)
        y_proba = model_pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        bal_accuracy = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        time_taken = time.time() - start_time

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Balanced Accuracy": bal_accuracy,
            "ROC AUC": roc_auc,
            "F1 Score": f1,
            "Time Taken": time_taken
        })

    results_df = pd.DataFrame(results)
    return results_df