# --- Imports ---
import os
import h5py
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# --- Functions ---
def generate_evaluation_table(results_df):
    """
    Generates a Markdown table from a DataFrame of model evaluation results.
    """

    markdown_table = "| Model                          | Accuracy  | Balanced Accuracy | ROC AUC   | F1 Score  | Time Taken |\n"
    markdown_table += "|:-------------------------------|----------:|------------------:|----------:|----------:|-----------:|\n"

    for index, row in results_df.iterrows():
        model_name = row['Model'].ljust(30)  # Adjust the number to fit your longest model name
        accuracy = f"{row['Accuracy']:.6f}".rjust(9)
        bal_accuracy = f"{row['Balanced Accuracy']:.6f}".rjust(17)
        roc_auc = f"{row['ROC AUC']:.6f}".rjust(9)
        f1_score = f"{row['F1 Score']:.6f}".rjust(9)
        time_taken = f"{row['Time Taken']:.6f}".rjust(10)

        markdown_table += f"| {model_name} | {accuracy} | {bal_accuracy} | {roc_auc} | {f1_score} | {time_taken} |\n"

    return markdown_table

def load_image_series(folder_path):
    """
    Loads and processes images from a given folder path, targeting .gif files.
    """
    if pd.isnull(folder_path) or not folder_path:
        print(f"Invalid folder path: {folder_path}")
        return None
    
    image_series = []
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return None
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.gif'):
            image_path = os.path.join(folder_path, filename)
            try:
                image = load_img(image_path, target_size=(224, 224))
                image = img_to_array(image)
                image = image / 255.0
                image_series.append(image)
            except Exception as e:
                print(f"Error loading image: {image_path} - {str(e)}")
    
    if not image_series:
        print(f"No valid images found in folder: {folder_path}")
        return None

    return np.array(image_series)

def process_images_and_save_to_hdf5(folder_path, hdf5_path, group_name):
    """
    This function processes images from a folder and saves them to an HDF5 file.
    Returns a list indicating successful processing for each series.
    """
    processed_series = []
    success_flags = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        folder_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        processed_series = list(executor.map(load_image_series, folder_paths))

    with h5py.File(hdf5_path, 'a') as hdf_file:
        for i, images in enumerate(processed_series):
            if images is not None and images.size > 0:  # Ensure images were successfully loaded
                dataset_name = f"{group_name}/{i}"
                hdf_file.create_dataset(dataset_name, data=images)
                success_flags.append(True)
            else:
                success_flags.append(False)

    return success_flags

def load_images_from_hdf5(hdf5_path, group_name):
    """
    This function loads processed image series from an HDF5 file.
    """
    processed_series = []
    with h5py.File(hdf5_path, 'r') as hdf_file:
        if group_name in hdf_file:
            for i in range(len(hdf_file[group_name])):
                processed_series.append(np.array(hdf_file[f"{group_name}/{i}"]))
    return processed_series

def get_number_of_datasets_in_group(hdf5_path, group_name):
    """Helper function to count the number of datasets in a specific group in the HDF5 file."""
    with h5py.File(hdf5_path, 'r') as hdf_file:
        if group_name in hdf_file:
            return len(hdf_file[group_name])
    return 0

def process_or_load_image_series(data, config, imagery_type, column_name):
    """
    This function processes or loads image series for a specific imagery type and updates the DataFrame accordingly.
    """
    imagery_folder_key = f"{imagery_type.lower()}_imagery_folder"
    folder_name = config['data'][imagery_folder_key]
    
    hdf5_path = os.path.join(config['data']['processed_dir'], 'launch_data_images.hdf5')
    raw_dir_path = os.path.join(config['data']['raw_satellite_dir'], folder_name)
    
    if not os.path.exists(hdf5_path):
        print(f"Processing images and saving to {hdf5_path}.")
        success_flags = process_images_and_save_to_hdf5(raw_dir_path, hdf5_path, imagery_type)
    else:
        print(f"Loading processed images from {hdf5_path} for {imagery_type}.")
        # If loading, assume all were successful
        success_flags = [True] * get_number_of_datasets_in_group(hdf5_path, imagery_type)

    # Extend success_flags to match DataFrame length, assuming False for any missing data
    extended_success_flags = success_flags + [False] * (len(data) - len(success_flags))
    
    # Update DataFrame with references or placeholders
    data[column_name] = [f"{imagery_type}/{i}" if flag else np.nan for i, flag in enumerate(extended_success_flags)]

    return data
