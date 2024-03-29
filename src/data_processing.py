# --- Imports ---
import os
import pandas as pd
import numpy as np

# --- Functions ---
def process_launch_stats_data(launch_stats_df):
  '''Processes launch statistics data.'''

  # Generate deep copy
  df = launch_stats_df.copy(deep=True)

  # Define the new column names
  launch_stats_column_names = {
      'Launch Vehicle':     'LAUNCH_VEHICLE',
      'Payload':            'PAYLOAD',
      'Date':               'DATE',
      'Start LCC':          'START_LCC_EVAL',
      'End LCC':            'END_LCC_EVAL',
      'Total Time':         'TOTAL_EVAL_TIME',
      'Count with':         'LCC_VIOLATION_COUNT',
      'Lightning Rule':     'LIGHTNING_RULE_VIOLATED',
      'Unnamed: 8':         'LIGHTNING_RULE_VIOLATION_COUNT',
      'Unnamed: 9':         'LIGHTNING_RULE_VIOLATION_DURATION',
      'Cumulus':            'CUMULUS_RULE_VIOLATED',
      'Unnamed: 11':        'CUMULUS_RULE_VIOLATION_COUNT',
      'Unnamed: 12':        'CUMULUS_RULE_VIOLATION_DURATION',
      'Unnamed: 13':        'ANVIL_2009_RULE_VIOLATED',
      'Anvil  (Pre 2009)':  'ANVIL_2009_RULE_VIOLATION_COUNT',
      'Unnamed: 15':        'ANVIL_2009_RULE_VIOLATION_DURATION',
      'Attached Anvil':     'ATTACHED_ANVIL_RULE_VIOLATED',
      'Unnamed: 17':        'ATTACHED_ANVIL_RULE_VIOLATION_COUNT',
      'Unnamed: 18':        'ATTACHED_ANVIL_RULE_VIOLATION_DURATION',
      'Detached Anvil':     'DETACHED_ANVIL_RULE_VIOLATED',
      'Unnamed: 20':        'DETACHED_ANVIL_RULE_VIOLATION_COUNT',
      'Unnamed: 21':        'DETACHED_ANVIL_RULE_VIOLATION_DURATION',
      'Debris':             'DEBRIS_RULE_VIOLATED',
      'Unnamed: 23':        'DEBRIS_RULE_VIOLATION_COUNT',
      'Unnamed: 24':        'DEBRIS_RULE_VIOLATION_DURATION',
      'Disturbed':          'DISTURBED_RULE_VIOLATED',
      'Unnamed: 26':        'DISTURBED_RULE_VIOLATION_COUNT',
      'Unnamed: 27':        'DISTURBED_RULE_VIOLATION_DURATION',
      'Thick':              'THICK_RULE_VIOLATED',
      'Unnamed: 29':        'THICK_RULE_VIOLATION_COUNT',
      'Unnamed: 30':        'THICK_RULE_VIOLATION_DURATION',
      'Smoke':              'SMOKE_RULE_VIOLATED',
      'Unnamed: 32':        'SMOKE_RULE_VIOLATION_COUNT',
      'Unnamed: 33':        'SMOKE_RULE_VIOLATION_DURATION',
      'Field Mill':         'FIELD_MILL_RULE_VIOLATED',
      'Unnamed: 35':        'FIELD_MILL_RULE_VIOLATION_COUNT',
      'Unnamed: 36':        'FIELD_MILL_RULE_VIOLATION_DURATION',
      'Good Sense':         'GOOD_SENSE_RULE_VIOLATED',
      'Unnamed: 38':        'GOOD_SENSE_RULE_VIOLATION_COUNT',
      'Unnamed: 39':        'GOOD_SENSE_RULE_VIOLATION_DURATION',
      'User WX Constraint': 'USER_WX_CONSTRAINT_RULE_VIOLATED',
      'Unnamed: 41':        'USER_WX_CONSTRAINT_RULE_VIOLATION_COUNT',
      'Unnamed: 42':        'USER_WX_CONSTRAINT_RULE_VIOLATION_DURATION',
      'Unnamed: 43':        'UNNAMED_43'
  }

  # Rename the columns
  df = df.rename(columns=launch_stats_column_names)

  # Drop the first row
  df = df.drop(0)

  # Drop 'UNNAMED_43' column
  df = df.drop(columns=['UNNAMED_43'])

  # Convert 'LAUNCH_VEHCILE' to string
  df['LAUNCH_VEHICLE'] = df['LAUNCH_VEHICLE'].astype(str)

  # Convert 'PAYLOAD' to string
  df['PAYLOAD'] = df['PAYLOAD'].astype(str)

  # Standardize string columns and trim whitespace
  df['LAUNCH_VEHICLE'] = df['LAUNCH_VEHICLE'].str.strip().str.upper()
  df['PAYLOAD'] = df['PAYLOAD'].str.strip().str.upper()

  # Convert 'DATE' to datetime
  df['DATE'] = pd.to_datetime(df['DATE'])

  # Convert START_LCC_EVAL and END_LCC_EVAL to datetime
  df["START_LCC_EVAL"] = pd.to_datetime(
      df["START_LCC_EVAL"], format="%m/%d/%y %H:%M", errors='coerce')
  df["END_LCC_EVAL"] = pd.to_datetime(
      df["END_LCC_EVAL"], format="%m/%d/%y %H:%M", errors='coerce')

  # --- Helper Function ---
  def adjust_end_eval_date(row):
      # Ensure START_LCC_EVAL and END_LCC_EVAL are in datetime format
      start_eval = pd.to_datetime(row['START_LCC_EVAL'])
      end_eval = pd.to_datetime(row['END_LCC_EVAL'])

      # Calculate the initial time difference
      time_difference = end_eval - start_eval

      # If the difference is negative (implying end_eval is before start_eval), adjust the date
      if time_difference.days < 0:
          # Add enough days to end_eval to make the time difference non-negative
          adjusted_days = abs(time_difference.days)
          end_eval = end_eval + pd.Timedelta(days=adjusted_days)
      elif time_difference.days > 0:
          # If the difference is more than a day, which shouldn't be the case, reduce it to within a day
          end_eval = end_eval - pd.Timedelta(days=time_difference.days)

      return end_eval

  # Apply the END_LCC_EVAL correction
  df['END_LCC_EVAL'] = df.apply(adjust_end_eval_date, axis=1)

  # Calculating the duration
  df["TOTAL_EVAL_TIME"] = df["END_LCC_EVAL"] - df["START_LCC_EVAL"]

  # Handling the conversion of 'LCC_VIOLATION_COUNT' and similar columns
  df['LCC_VIOLATION_COUNT'] = pd.to_numeric(df['LCC_VIOLATION_COUNT'], errors='coerce').fillna(0).astype(int)

  # Handling columns ending with 'RULE_VIOLATED' to be boolean
  for col in df.columns:
      if 'RULE_VIOLATED' in col:
          df[col] = df[col].notnull()

  # Converting 'VIOLATION_COUNT' columns to integer
  for col in df.columns:
      if 'VIOLATION_COUNT' in col:
          df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

  # Handling 'VIOLATION_DURATION' conversion to timedelta
  for col in df.columns:
      if 'VIOLATION_DURATION' in col:
          # Convert to string and ensure every entry has at least one ':' delimiter
          df[col] = df[col].apply(lambda x: str(x) if ':' in str(x) else "0:0")
          # Split the column into two parts: hours and minutes, ensuring conversion to string first
          hours_minutes = df[col].str.split(':', expand=True)

          # Now, convert these parts to numeric, treating non-numeric as NaN (which will then be filled with 0)
          hours_minutes[0] = pd.to_numeric(hours_minutes[0], errors='coerce').fillna(0).astype(int)
          hours_minutes[1] = pd.to_numeric(hours_minutes[1], errors='coerce').fillna(0).astype(int)

          # Convert to timedelta
          df[col] = pd.to_timedelta(hours_minutes[0], unit='h') + pd.to_timedelta(hours_minutes[1], unit='m')

  return df

def process_launch_forecast_data(launch_forecast_df):
  '''Processes launch forecast data.'''

  # Generate deep copy
  df = launch_forecast_df.copy(deep=True)

  # Rename columns to uppercase and replace spaces with underscores
  df.columns = df.columns.str.upper().str.replace(' ', '_')

  # Rename specific columns
  df = df.rename(columns={
      'L_-_4_DAY_FCST': 'L4_DAY_FORECAST',
      'L_-_3_DAY_FCST': 'L3_DAY_FORECAST',
      'L_-_2_DAY_FCST': 'L2_DAY_FORECAST',
      'L_-_1_DAY_FCST': 'L1_DAY_FORECAST'
  })

  # Drop columns after 'LAUNCHED'
  cols_to_drop = df.columns[df.columns.get_loc('LAUNCHED')+1:]
  df = df.drop(columns=cols_to_drop)

  # Standardize string columns and trim whitespace
  df['LAUNCH_VEHICLE'] = df['LAUNCH_VEHICLE'].str.strip().str.upper()
  df['PAYLOAD'] = df['PAYLOAD'].str.strip().str.upper()

  # Convert LAUNCH_DATE to datetime
  df["LAUNCH_DATE"] = pd.to_datetime(df["LAUNCH_DATE"])

  # Define a mapping for boolean conversion
  bool_map = {'yes': True, 'no': False, '0.0': False, '1': True, '1.0': True, '0': False}

  # Standardize and convert specific columns to boolean
  for col in ['RANGE_LCC_WX_VIOLATION_OCCURRED', 'USER_WX_LCC_VIOLATION_OCCURRED', 'RANGE_OR_USER_VIOLATION', 'LAUNCHED']:
      # Ensure the data is in a consistent format (as string), NaNs will be untouched
      df[col] = df[col].astype(str).str.lower()

      # Apply the mapping
      df[col] = df[col].replace(bool_map)

      # Replace 'nan' string (from the astype(str) conversion of NaN values) with actual NaN, then fill with False
      df[col] = df[col].replace('nan', np.nan).fillna(False)

  return df

def process_weather_hourly_data(weather_hourly_df):
  """
  Processes the weather_hourly_df DataFrame to ensure correct data types.
  """

  # Generate deep copy
  df = weather_hourly_df.copy(deep=True)

  # Preprocess 'dt_iso' to remove timezone information
  df['dt_iso'] = df['dt_iso'].str.replace(r'\s+\+\d+\s+UTC', '', regex=True)

  # Convert 'dt_iso' to datetime
  df['dt_iso'] = pd.to_datetime(df['dt_iso'], errors='coerce')

  # Convert 'dt' to a datetime format for easier manipulation (optional, based on need)
  df['dt'] = pd.to_datetime(df['dt'], unit='s', utc=True).dt.tz_convert(None)

  return df

def apply_manual_corrections(clean_launch_stats_df, clean_launch_forecast_df):
    """
    Manually applies corrections to reconcile differences in payload names and dates
    between the launch stats and forecast dataframes.
    """

    # Generate deep copies
    forecast_df = clean_launch_forecast_df.copy(deep=True)
    stats_df = clean_launch_stats_df.copy(deep=True)

    # Corrections for PAYLOAD names
    payload_corrections = {
      'TELSTAR 19V': 'TELSTAR-19V',
      'TELSTAR 18V': 'TELSTAR-18V',
      'DRAGON IFA': 'DRAGON-IFA',
      'K-MILSAT 1': 'K-MILSAT-1',
      'STARLINK-3': 'STARLINK V1 L3',
      'STARLINK-4': 'STARLINK V1 L4',
      'STARLINK-7': 'STARLINK V1 L7',
      'STARLINK-8': 'STARLINK V1 L8',
      'GPS III-F2': 'GPS III SV02',
      'GPS-III-3': 'GPS III SV03',
      'GPS III-5': 'GPS III SV05',
      'CRS-18': 'SPX-18',
      'CREW DEMO 1': 'CREW DRAGON DEMO-1',
      'DRAGON IFA': 'IN-FLIGHT ABORT TEST',
      'GLOBALSTAR-FN15': 'GLOBALSTAR FM-15',
      'GLOBALSTAR FN-15': 'GLOBALSTAR FM-15'
    }

    # Apply PAYLOAD name corrections
    stats_df['PAYLOAD'] = stats_df['PAYLOAD'].replace(payload_corrections)
    forecast_df['PAYLOAD'] = forecast_df['PAYLOAD'].replace(payload_corrections)

    # Correct the 'LAUNCH_VEHICLE' for TROPICS-1
    stats_df.loc[stats_df['PAYLOAD'] == 'TROPICS-1', 'LAUNCH_VEHICLE'] = 'ASTRA'
    forecast_df.loc[forecast_df['PAYLOAD'] == 'TROPICS-1', 'LAUNCH_VEHICLE'] = 'ASTRA'

    # Manual date corrections based on specific PAYLOAD matches
    date_corrections = {
        ('DELTA IV', 'WGS-10', '2019-03-15'): '2019-03-16',
        ('FALCON 9', 'STARLINK', '2019-05-24'): '2019-05-23',
        ('FALCON 9', 'TROPICS-1', '2022-06-12'): '2022-06-12',  # Corrects the LAUNCH_VEHICLE for TROPICS-1 as well
        ('FALCON 9', 'GLOBALSTAR FM15', '2022-06-19'): '2022-06-19',
    }

    for key, new_date in date_corrections.items():
        vehicle, payload, old_date = key
        # Update in launch forecast dataframe
        mask = (forecast_df['LAUNCH_VEHICLE'] == vehicle) & (forecast_df['PAYLOAD'] == payload) & (forecast_df['LAUNCH_DATE'].astype(str) == old_date)
        forecast_df.loc[mask, 'LAUNCH_DATE'] = new_date

    # Correct the 'LAUNCH_VEHICLE' for TROPICS-1 based on PAYLOAD
    forecast_df.loc[forecast_df['PAYLOAD'] == 'TROPICS-1', 'LAUNCH_VEHICLE'] = 'FALCON 9'

    return stats_df, forecast_df

def add_image_data_to_dataset(clean_launch_stats_df, goes_visible_folder_path, effective_shear_folder_path, watch_warning_folder_path, sbcape_cin_folder_path):
    """
    Enhances the launch_data DataFrame with paths to image data based on the END_LCC_EVAL date.

    Parameters:
    - launch_data: DataFrame containing the launch data.
    - goes_visible_folder_path: Base path for GOES Visible images.
    - effective_shear_folder_path: Base path for Effective Shear images.
    - watch_warning_folder_path: Base path for Watch Warning images.
    - sbcape_cin_folder_path: Base path for SBCAPE CIN images.

    Returns:
    - Enhanced DataFrame with image data paths.
    """

    # Convert END_LCC_EVAL to string in YYYYMMDD format
    clean_launch_stats_df['END_LCC_EVAL_STR'] = clean_launch_stats_df['END_LCC_EVAL'].dt.strftime('%Y%m%d')

    # Define a function to check for folder and contents
    def folder_path_with_content(base_path, folder_name):
        full_path = os.path.join(base_path, folder_name)
        if os.path.isdir(full_path) and os.listdir(full_path):
            return full_path
        return ""

    # Apply the function to get paths for each of the four types of images
    clean_launch_stats_df['GOES_IMG_PATH'] = clean_launch_stats_df['END_LCC_EVAL_STR'].apply(lambda x: folder_path_with_content(goes_visible_folder_path, x))
    clean_launch_stats_df['SHEAR_IMG_PATH'] = clean_launch_stats_df['END_LCC_EVAL_STR'].apply(lambda x: folder_path_with_content(effective_shear_folder_path, x))
    clean_launch_stats_df['WARNIGN_IMG_PATH'] = clean_launch_stats_df['END_LCC_EVAL_STR'].apply(lambda x: folder_path_with_content(watch_warning_folder_path, x))
    clean_launch_stats_df['SBCAPE_CIN_IMG_PATH'] = clean_launch_stats_df['END_LCC_EVAL_STR'].apply(lambda x: folder_path_with_content(sbcape_cin_folder_path, x))

    # Drop the temporary column
    clean_launch_stats_df.drop('END_LCC_EVAL_STR', axis=1, inplace=True)

    return clean_launch_stats_df

def save_datasets(clean_launch_stats_df, clean_launch_forecast_df):
    '''Saves the processed datasets to the data directory.'''

    # Save the processed datasets
    clean_launch_stats_df.to_csv("data/processed/clean_launch_stats.csv", index=False)
    clean_launch_forecast_df.to_csv("data/processed/clean_launch_forecast.csv", index=False)
