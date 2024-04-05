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
    one_hot_encode_categorical_columns,
    convert_bool_to_int
)
from modeling import (
    assign_modeling_roles, 
    prepare_image_data,
    define_and_compile_cnn_model,
    define_gradient_boosting_model,
    train_and_evaluate_cnn_model,
    train_and_evaluate_gb_model,
    ensemble_predictions
)
from utils import generate_evaluation_table, integrate_image_data

# --- Imports ---
import yaml
import pandas as pd

# --- Load Data ---
print("\nLoading data...")
print("--------------------")

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
print("\nProcessing data...")
print("--------------------")

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
launch_data = convert_bool_to_int(launch_data)
launch_data = integrate_image_data(launch_data, config)


# --- Modeling ---
print("\nModeling...")
print("--------------------")

# Define path to HDF5 file and columns with image references
hdf5_path = f"{config['data']['processed_dir']}/{config['data']['hdf5_file']}"
image_series_columns = ['GOES_REF', 'SHEAR_REF', 'WARNING_REF', 'SBCAPE_CIN_REF']

# Assign roles to the features and target for modeling
print("\nAssigning roles to the features and target for modeling...")
X_train, X_test, y_train, y_test = assign_modeling_roles(launch_data, hdf5_path)
X_image_train, X_image_test = prepare_image_data(X_train, X_test, image_series_columns, hdf5_path)

# Define and compile the CNN model
print("\nDefining and compiling the CNN model...")
cnn_model = define_and_compile_cnn_model()

# Train and evaluate the CNN model
print("\nTraining and evaluating the CNN model...")
cnn_metrics, y_pred_cnn = train_and_evaluate_cnn_model(cnn_model, X_image_train, X_image_test, y_train, y_test)

# Define and train the Gradient Boosting model
print("\nDefining the Gradient Boosting model...")
gb_model = define_gradient_boosting_model()

# Train and evaluate the Gradient Boosting model
print("\nTraining and evaluating the Gradient Boosting model...")
gb_metrics, y_pred_gb = train_and_evaluate_gb_model(gb_model, X_train, X_test, y_train, y_test)

# Ensemble the predictions
print("\nEnsembling the predictions...")
y_pred_ensemble = ensemble_predictions(y_pred_cnn, y_pred_gb)

# --- Evaluation ---
print("\nEvaluation...")
print("--------------------")

# Generate the evaluation table
results_df = pd.DataFrame([
    {'Model': 'CNN', **cnn_metrics},
    {'Model': 'Gradient Boosting', **gb_metrics}
])

markdown_table = generate_evaluation_table(results_df)
print(markdown_table)