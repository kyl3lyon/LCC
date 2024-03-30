# --- Module Level Imports ---
from feature_engineering import load_image_series

# --- Imports ---
import os
import h5py
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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

def process_images_and_save_to_hdf5(folder_path, hdf5_path, group_name):
    """
    Processes images from a folder and saves them to an HDF5 file.
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

def load_images_from_hdf5(hdf5_path, group_name, data_length):
    """
    Loads processed image series from an HDF5 file for a specific group.
    If the group does not exist, it fills the corresponding DataFrame column with np.nan.
    """
    processed_series = []
    with h5py.File(hdf5_path, 'r') as hdf_file:
        if group_name in hdf_file:
            # If the group exists, load all image references
            processed_series = [f"{group_name}/{i}" for i in range(len(hdf_file[group_name]))]
        else:
            # If the group does not exist, fill with np.nan
            processed_series = [np.nan] * data_length  # Ensure length matches the DataFrame
    
    return processed_series

def get_number_of_datasets_in_group(hdf5_path, group_name):
    """
    Helper function to count the number of datasets in a specific group in the HDF5 file.
    """
    with h5py.File(hdf5_path, 'r') as hdf_file:
        if group_name in hdf_file:
            return len(hdf_file[group_name])
    return 0

def process_or_load_image_series(data, config, imagery_type, column_name):
    """
    Processes or loads image series for a specific imagery type and column name.
    """
    hdf5_path = os.path.join(config['data']['processed_dir'], 'launch_data_images.hdf5')
    
    # Process images and save to HDF5 if not exists
    if not os.path.exists(hdf5_path):
        print(f"HDF5 file not found at {hdf5_path}. Attempting to process images.")
        folder_path = os.path.join(config['data']['raw_satellite_dir'], config['data'][f"{imagery_type.lower()}_imagery_folder"])
        success_flags = process_images_and_save_to_hdf5(folder_path, hdf5_path, imagery_type.lower())
        # After processing, ensure each row in DataFrame has a placeholder
        data[column_name] = [f"{imagery_type.lower()}/{i}" if flag else np.nan for i, flag in enumerate(success_flags)] + [np.nan] * (len(data) - len(success_flags))
    else:
        with h5py.File(hdf5_path, 'r') as hdf_file:
            if imagery_type.lower() in hdf_file:
                # Load data if the group exists
                group_len = len(hdf_file[imagery_type.lower()])
                # Ensure the column is filled out to match DataFrame's length
                references = [f"{imagery_type.lower()}/{i}" for i in range(group_len)]
                data[column_name] = references + [np.nan] * (len(data) - group_len)
            else:
                # If the group doesn't exist, fill the column with np.nan
                print(f"Group for {imagery_type} not found. Filling {column_name} with np.nan.")
                data[column_name] = [np.nan] * len(data)

    # This ensures the DataFrame column has an entry for each row
    assert len(data[column_name]) == len(data), "Column length does not match DataFrame length."

    return data

def integrate_image_data(launch_data, config):
    hdf5_path = os.path.join(config['data']['processed_dir'], 'launch_data_images.hdf5')
    
    imagery_types = ['GOES', 'SHEAR', 'WARNING', 'SBCAPE_CIN']
    
    # Check if the HDF5 file exists
    if not os.path.exists(hdf5_path):
        print(f"HDF5 file not found at {hdf5_path}. Attempting to process images.")
        
        # Process images for all imagery types and save to HDF5
        for imagery_type in imagery_types:
            column_name = f"{imagery_type}_REF"  # Column for references
            folder_path = os.path.join(config['data']['raw_satellite_dir'], config['data'][f"{imagery_type.lower()}_imagery_folder"])
            success_flags = process_images_and_save_to_hdf5(folder_path, hdf5_path, imagery_type.lower())
            
            # After processing, ensure each row in DataFrame has a placeholder
            launch_data[column_name] = [f"{imagery_type.lower()}/{i}" if flag else np.nan for i, flag in enumerate(success_flags)] + [np.nan] * (len(launch_data) - len(success_flags))
    
    else:
        print(f"HDF5 file found at {hdf5_path}. Loading image references.")
        
        # Load image references for all imagery types from HDF5
        for imagery_type in imagery_types:
            column_name = f"{imagery_type}_REF"  # Column for references
            
            with h5py.File(hdf5_path, 'r') as hdf_file:
                if imagery_type.lower() in hdf_file:
                    # Load data if the group exists
                    group_len = len(hdf_file[imagery_type.lower()])
                    # Ensure the column is filled out to match DataFrame's length
                    references = [f"{imagery_type.lower()}/{i}" for i in range(group_len)]
                    launch_data[column_name] = references + [np.nan] * (len(launch_data) - group_len)
                else:
                    # If the group doesn't exist, fill the column with np.nan
                    print(f"Group for {imagery_type} not found. Filling {column_name} with np.nan.")
                    launch_data[column_name] = [np.nan] * len(launch_data)
    
    # Ensure each DataFrame column has an entry for each row
    for imagery_type in imagery_types:
        column_name = f"{imagery_type}_REF"
        assert len(launch_data[column_name]) == len(launch_data), f"Column length for {column_name} does not match DataFrame length."
    
    return launch_data