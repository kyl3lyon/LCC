# --- Imports ---
import pandas as pd

# --- Functions ---
def filter_weather_for_launch(weather_df, start_time, end_time):
    """
    Filters weather data for a given time window.
    """
    return weather_df[(weather_df['dt_iso'] >= start_time) & (weather_df['dt_iso'] <= end_time)]

def calculate_average_weather(filtered_weather_df):
    """
    Calculates the average of weather parameters in the filtered weather DataFrame.
    """
    if filtered_weather_df.empty:
        return pd.Series({
            'visibility': None, 'dew_point': None, 'feels_like': None,
            'temp_min': None, 'temp_max': None, 'pressure': None, 'humidity': None,
            'wind_speed': None, 'wind_deg': None, 'clouds_all': None, 'rain_1h': None,
            'rain_3h': None, 'weather_main': None, 'weather_description': None,
            'weather_icon': None
        })

    averages = filtered_weather_df[['visibility', 'dew_point', 'feels_like', 'temp_min', 'temp_max',
                                    'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all',
                                    'rain_1h', 'rain_3h']].mean()

    # Handling cases where there might not be a mode
    most_common_main = filtered_weather_df['weather_main'].mode()
    most_common_description = filtered_weather_df['weather_description'].mode()
    most_common_icon = filtered_weather_df['weather_icon'].mode()

    averages['weather_main'] = most_common_main.iloc[0] if not most_common_main.empty else None
    averages['weather_description'] = most_common_description.iloc[0] if not most_common_description.empty else None
    averages['weather_icon'] = most_common_icon.iloc[0] if not most_common_icon.empty else None

    return averages

def aggregate_weather_for_launches(weather_df, launch_df):
    """
    Aggregates weather data for each launch, calculating average weather conditions during the launch window.
    """
    averages_list = []

    for index, launch in launch_df.iterrows():
        # Filter weather data for the current launch's time window
        filtered_weather = filter_weather_for_launch(weather_df, launch['START_LCC_EVAL'], launch['END_LCC_EVAL'])

        # Calculate average weather parameters for the filtered data, including most frequent weather_main and description
        averages = calculate_average_weather(filtered_weather)

        # Add identifiers from launch record
        for col in launch.index:
            averages[col] = launch[col]

        # Append the averages to the list
        averages_list.append(averages)

    # Convert the list to a DataFrame
    averages_df = pd.DataFrame(averages_list)

    return averages_df

def join_launch_stats_weather_with_actuals(launch_stats_and_weather_df, clean_launch_forecast_df, return_non_matching_columns=False):
    """
    Joins launch stats and weather data with launch forecast data on specified keys.
    """

    # Adjusting column types to string for merging, since direct date comparison can be tricky with pandas dtypes
    launch_stats_and_weather_df['DATE'] = launch_stats_and_weather_df['DATE'].astype(str)
    clean_launch_forecast_df['LAUNCH_DATE'] = clean_launch_forecast_df['LAUNCH_DATE'].astype(str)

    # Perform the join operation based on dates
    merged_df = pd.merge(
        launch_stats_and_weather_df,
        clean_launch_forecast_df,
        left_on=['LAUNCH_VEHICLE', 'PAYLOAD', 'DATE'],
        right_on=['LAUNCH_VEHICLE', 'PAYLOAD', 'LAUNCH_DATE'],
        how='inner'
    )

    if return_non_matching_columns:
      # Identify columns that didn't match from both sides
      left_cols = set(launch_stats_and_weather_df.columns)
      right_cols = set(clean_launch_forecast_df.columns)
      merged_cols = set(merged_df.columns)

      non_matching_left = left_cols - merged_cols
      non_matching_right = right_cols - merged_cols
      non_matching_columns = list(non_matching_left.union(non_matching_right))

      return merged_df, non_matching_columns

    return merged_df

def find_missing_rows_after_join(launch_stats_and_weather_df, clean_launch_forecast_df):
    """
    Finds rows in launch_stats_and_weather_df and clean_launch_forecast_df that do not match based on specified keys.
    """
    # Ensure the key columns are of the same data type for accurate comparison
    launch_stats_and_weather_df['DATE'] = launch_stats_and_weather_df['DATE'].astype(str)
    clean_launch_forecast_df['LAUNCH_DATE'] = clean_launch_forecast_df['LAUNCH_DATE'].astype(str)

    # Perform an outer join with an indicator to find unmatched rows
    merged_outer = pd.merge(
        launch_stats_and_weather_df,
        clean_launch_forecast_df,
        left_on=['LAUNCH_VEHICLE', 'PAYLOAD', 'DATE'],
        right_on=['LAUNCH_VEHICLE', 'PAYLOAD', 'LAUNCH_DATE'],
        how='outer',
        indicator=True
    )

    # Rows present only in the first DataFrame (launch_stats_and_weather_df)
    only_in_first = merged_outer[merged_outer['_merge'] == 'left_only']

    # Rows present only in the second DataFrame (clean_launch_forecast_df)
    only_in_second = merged_outer[merged_outer['_merge'] == 'right_only']

    return only_in_first, only_in_second