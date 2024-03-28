# --- Module Level Imports ---
from data_processing import (
    process_launch_stats_data,
    process_launch_forecast_data,
    process_weather_hourly_data,
    apply_manual_corrections,
    save_datasets
)
from feature_engineering import aggregate_weather_for_launches, join_launch_stats_weather_with_actuals
from modeling import assign_modeling_roles, evaluate_models
from utils import generate_evaluation_table


# --- Imports ---
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# --- Load Data ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

launch_stats_df = pd.read_csv(f"{config['data']['raw_dir']}/{config['data']['launch_stats_file']}")
launch_forecast_df = pd.read_csv(f"{config['data']['raw_dir']}/{config['data']['launch_forecast_file']}")
weather_hourly_df = pd.read_csv(f"{config['data']['raw_dir']}/{config['data']['weather_hourly_file']}")

# --- Data Processing ---
clean_launch_stats_df = process_launch_stats_data(launch_stats_df)
clean_launch_forecast_df = process_launch_forecast_data(launch_forecast_df)
clean_weather_hourly_df = process_weather_hourly_data(weather_hourly_df)
clean_launch_stats_df, clean_launch_forecast_df = apply_manual_corrections(clean_launch_stats_df, clean_launch_forecast_df)
save_datasets(clean_launch_stats_df, clean_launch_forecast_df)

# --- Feature Engineering ---
launch_stats_and_weather_df = aggregate_weather_for_launches(clean_weather_hourly_df, clean_launch_stats_df)
launch_data = join_launch_stats_weather_with_actuals(launch_stats_and_weather_df, clean_launch_forecast_df)

# --- Modeling ---
X_train, X_test, y_train, y_test = assign_modeling_roles(launch_data)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
}

results_df = evaluate_models(models, X_train, X_test, y_train, y_test)

# --- Evaluation ---
markdown_table = generate_evaluation_table(results_df)
print(markdown_table)