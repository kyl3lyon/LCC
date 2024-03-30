# --- Module Level Imports ---
from data_processing import (
    process_launch_stats_data,
    process_launch_forecast_data,
    process_weather_hourly_data,
    apply_manual_corrections,
    add_image_data_to_dataset,
    save_datasets
)
from feature_engineering import (
    aggregate_weather_for_launches,
    join_launch_stats_weather_with_actuals,
    one_hot_encode_categorical_columns
)
from modeling import (assign_modeling_roles, 
                      prepare_image_data, 
                      train_and_evaluate_model
)
from utils import generate_evaluation_table, integrate_image_data

# --- Imports ---
import yaml
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# --- Load Data ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

launch_stats_df = pd.read_csv(f"{config['data']['raw_dir']}/{config['data']['launch_stats_file']}")
launch_forecast_df = pd.read_csv(f"{config['data']['raw_dir']}/{config['data']['launch_forecast_file']}")
weather_hourly_df = pd.read_csv(f"{config['data']['raw_dir']}/{config['data']['weather_hourly_file']}")
goes_visible_folder_path = f"{config['data']['raw_satellite_dir']}/{config['data']['goes_imagery_folder']}"
effective_shear_folder_path = f"{config['data']['raw_satellite_dir']}/{config['data']['shear_imagery_folder']}"
watch_warning_folder_path = f"{config['data']['raw_satellite_dir']}/{config['data']['warning_imagery_folder']}"
sbcape_cin_folder_path = f"{config['data']['raw_satellite_dir']}/{config['data']['sbcape_cin_imagery_folder']}"

# --- Data Processing ---
clean_launch_stats_df = process_launch_stats_data(launch_stats_df)
clean_launch_forecast_df = process_launch_forecast_data(launch_forecast_df)
clean_weather_hourly_df = process_weather_hourly_data(weather_hourly_df)
clean_launch_stats_df, clean_launch_forecast_df = apply_manual_corrections(clean_launch_stats_df, clean_launch_forecast_df)

clean_launch_stats_df = add_image_data_to_dataset(clean_launch_stats_df, goes_visible_folder_path,
                                                  effective_shear_folder_path, watch_warning_folder_path,
                                                  sbcape_cin_folder_path)

save_datasets(clean_launch_stats_df, clean_launch_forecast_df)

# --- Feature Engineering ---
launch_stats_and_weather_df = aggregate_weather_for_launches(clean_weather_hourly_df, clean_launch_stats_df)
launch_data = join_launch_stats_weather_with_actuals(launch_stats_and_weather_df, clean_launch_forecast_df)
launch_data = one_hot_encode_categorical_columns(launch_data)
launch_data = integrate_image_data(launch_data, config)


# --- Modeling ---

# Define path to HDF5 file and columns with image references
hdf5_path = f"{config['data']['processed_dir']}/{config['data']['hdf5_file']}"
image_series_columns = ['GOES_REF', 'SHEAR_REF', 'WARNING_REF', 'SBCAPE_CIN_REF']

# Assign roles to the features and target for modeling
X_train, X_test, y_train, y_test = assign_modeling_roles(launch_data, hdf5_path)
X_image_train, X_image_test = prepare_image_data(X_train, X_test, image_series_columns, hdf5_path)

# Define and train the CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and evaluate the CNN model
cnn_metrics = train_and_evaluate_model(cnn_model, X_train, X_test, y_train, y_test, X_image_train, X_image_test)

# Define and train the Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train and evaluate the Gradient Boosting model
gb_metrics = train_and_evaluate_model(gb_model, X_train, X_test, y_train, y_test, X_image_train, X_image_test)

# --- Evaluation ---
# Generate the evaluation table
results_df = pd.DataFrame([
    {'Model': 'CNN', **cnn_metrics},
    {'Model': 'Gradient Boosting', **gb_metrics}
])

markdown_table = generate_evaluation_table(results_df)
print(markdown_table)