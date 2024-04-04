# --- Imports ---
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import h5py
import time

# --- Preprocessor Definition ---

# Define the numeric features
numeric_features = ['VISIBILITY', 'DEW_POINT', 'FEELS_LIKE', 'TEMP_MIN', 'TEMP_MAX',
                    'PRESSURE', 'HUMIDITY', 'WIND_SPEED', 'WIND_DEG', 'CLOUDS_ALL',
                    'RAIN_1H', 'RAIN_3H'
]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Define the categorical features
categorical_features = [
    'CLEAR', 'CLOUDS', 'DRIZZLE', 'FOG', 'HAZE', 'MIST', 'RAIN', 
    'BROKEN_CLOUDS', 'FEW_CLOUDS', 'HEAVY_INTENSITY_RAIN', 
    'LIGHT_INTENSITY_DRIZZLE', 'LIGHT_RAIN', 'MODERATE_RAIN', 
    'OVERCAST_CLOUDS', 'SCATTERED_CLOUDS', 'SKY_IS_CLEAR',
    '01d', '01n', '02d', '02n', '03d', '03n', '04d', '04n',
    '09d', '09n', '10d', '10n', '50d', '50n'
]
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))])

# Define the image features
image_series_columns = ['GOES_REF', 'SHEAR_REF', 'WARNING_REF', 'SBCAPE_CIN_REF']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# --- Functions ---
def load_series_length(hdf5_path, references):
    """Load the length of image series from HDF5 file based on provided references."""
    lengths = []
    with h5py.File(hdf5_path, 'r') as hdf_file:
        for ref in references:
            try:
                lengths.append(len(hdf_file[ref][:]))
            except KeyError:
                # Reference not found in HDF5, assuming missing series
                lengths.append(0)
    return np.array(lengths)

def load_image_data(hdf5_path, references):
    """
    Loads image series data from an HDF5 file based on provided references.
    """
    images_data = []
    with h5py.File(hdf5_path, 'r') as hdf_file:
        for ref in references:
            try:
                images_data.append(hdf_file[ref][:])
            except KeyError:
                # Reference not found, append a placeholder (e.g., an array of zeros)
                images_data.append(np.zeros((7, 224, 224, 3)))  # Adjust shape as necessary
    return np.array(images_data)

def prepare_image_data(X_train, X_test, image_series_columns, hdf5_path):
    """
    Loads image data for training and testing sets based on references in the DataFrame.

    Parameters:
    - X_train, X_test: DataFrames containing the training and testing splits.
    - image_series_columns: List of columns in the DataFrame that contain references to the image data.
    - hdf5_path: Path to the HDF5 file containing the image data.

    Returns:
    - X_image_train, X_image_test: Dictionaries containing the loaded image data arrays for training and testing.
    """
    X_image_train = {}
    X_image_test = {}

    for column in image_series_columns:
        train_refs = X_train[column].dropna().tolist()
        test_refs = X_test[column].dropna().tolist()
        
        X_image_train[column] = load_image_data(hdf5_path, train_refs)
        X_image_test[column] = load_image_data(hdf5_path, test_refs)
    
    return X_image_train, X_image_test

def assign_modeling_roles(launch_data, hdf5_path):
    """
    Assigns roles to the features and target for modeling.
    """
    # Initialize valid_indices as a Series of True values for each row in launch_data
    valid_indices = pd.Series(True, index=launch_data.index)

    # Filter launch_data based on the existence and length of image series
    for column in image_series_columns:
        references = launch_data[column].dropna().tolist()  # Get the references and drop missing values
        series_lengths = load_series_length(hdf5_path, references)  # Load the series lengths from HDF5
        column_valid_indices = series_lengths == 7  # Exactly 7 images in the series
        valid_indices &= launch_data[column].index.isin(np.where(column_valid_indices)[0])  # Filter based on valid indices

    # Filter the launch_data based on the valid indices
    launch_data = launch_data[valid_indices]

    # Define your predictors and target
    target = 'LAUNCHED'  # Update with your actual target column name
    predictors = numeric_features + categorical_features + image_series_columns

    X = launch_data[predictors]
    y = launch_data[target]

    # Splitting the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train.astype('int'), y_test.astype('int')

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, X_image_train, X_image_test):
    """
    Trains and evaluates a model on the provided data.

    Parameters:
    - model: The model to train and evaluate.
    - X_train, X_test: DataFrames containing the training and testing features.
    - y_train, y_test: Arrays containing the training and testing labels.
    - X_image_train, X_image_test: Dictionaries containing the image data for training and testing.

    Returns:
    - A dictionary containing the evaluation metrics.
    """
    start_time = time.time()
    
    # Fit the preprocessor on the training data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    
    # Transform the test data using the fitted preprocessor
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Train the model
    model.fit(X_train_preprocessed, y_train)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test_preprocessed)
    
    end_time = time.time()
    
    # Calculate evaluation metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'Time Taken': end_time - start_time
    }
    
    return metrics