"""
Weather Data Fetcher for Environment Canada Climate Daily API

This module fetches weather data for Toronto/Ontario to support hotel booking analysis.
Migrated from Google Colab to PyCharm environment.
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime

def get_weather_data_paginated(province_code, local_year, limit=500):
    """
    Fetches all weather data from the Environment Canada Climate Daily API
    for a given province and year using pagination.

    Args:
        province_code (str): The two-letter province code (e.g., 'ON', 'QC').
        local_year (int): The year for which to fetch data.
        limit (int): The maximum number of records to fetch per request.

    Returns:
        pandas.DataFrame: A DataFrame containing all the weather data for the specified year and province, or None if an error occurred.
    """
    base_api_url = "https://api.weather.gc.ca/collections/climate-daily/items"
    all_data_list = []
    offset = 0

    print(f"Fetching paginated data for {province_code}, year {local_year} with limit {limit}...")

    while True:
        params = {
            'limit': limit,
            'offset': offset,
            'PROVINCE_CODE': province_code,
            'LOCAL_YEAR': local_year
        }

        try:
            response = requests.get(base_api_url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'features' in data:
                features = data['features']
                if not features:
                    # No more data to fetch
                    print(f"  No more data for {local_year} at offset {offset}.")
                    break

                properties_list = [feature['properties'] for feature in features]
                all_data_list.extend(properties_list)
                print(f"  Fetched {len(properties_list)} records for {local_year} at offset {offset}. Total records fetched so far: {len(all_data_list)}")

                # Check if the number of returned records is less than the limit, indicating the last page
                if data.get('numberReturned', limit) < limit:
                     print(f"  Less than {limit} records returned, assuming last page for {local_year}.")
                     break


                offset += limit
                # Add a small delay to avoid overwhelming the API
                time.sleep(1)

            else:
                print("Unexpected JSON structure: 'features' key not found.")
                print("API response keys:", data.keys())
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from API: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    if all_data_list:
        return pd.DataFrame(all_data_list)
    else:
        print(f"No data fetched for {province_code}, year {local_year}.")
        return None

def fetch_weather_data_for_years(years_list, province_code='ON'):
    """
    Fetch weather data for multiple years and combine into a single DataFrame.
    
    Args:
        years_list (list): List of years to fetch data for
        province_code (str): Province code (default: 'ON' for Ontario)
    
    Returns:
        pandas.DataFrame: Combined weather data for all years
    """
    all_years_df_list = []
    
    print(f"\n🌤️ Fetching weather data for {province_code} for years: {years_list}")
    
    for year in years_list:
        print(f"\n📅 Processing year {year}...")
        df_year = get_weather_data_paginated(province_code, year, limit=500)
        
        if df_year is not None:
            all_years_df_list.append(df_year)
            print(f"✅ Finished fetching data for {year}. Records: {len(df_year)}")
        else:
            print(f"❌ Failed to fetch data for {year}")
    
    if all_years_df_list:
        combined_df = pd.concat(all_years_df_list, ignore_index=True)
        print(f"\n🎉 Combined weather data shape: {combined_df.shape}")
        return combined_df
    else:
        print("\n❌ No weather data was successfully fetched")
        return None

def clean_weather_data(weather_df):
    """
    Clean and process weather data by selecting relevant columns and handling data types.
    
    Args:
        weather_df (pandas.DataFrame): Raw weather data
    
    Returns:
        pandas.DataFrame: Cleaned weather data
    """
    if weather_df is None or weather_df.empty:
        print("❌ No weather data provided for cleaning")
        return None
    
    print("🧹 Cleaning weather data...")
    
    # Select relevant columns for hotel booking analysis
    columns_to_keep = [
        'STATION_NAME', 'CLIMATE_IDENTIFIER', 'ID', 'LOCAL_DATE', 
        'PROVINCE_CODE', 'LOCAL_YEAR', 'LOCAL_MONTH', 'LOCAL_DAY',
        'MEAN_TEMPERATURE', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE',
        'TOTAL_PRECIPITATION', 'TOTAL_RAIN', 'TOTAL_SNOW', 
        'SNOW_ON_GROUND', 'SPEED_MAX_GUST', 'MAX_REL_HUMIDITY'
    ]
    
    # Keep only columns that exist in the DataFrame
    available_columns = [col for col in columns_to_keep if col in weather_df.columns]
    cleaned_df = weather_df[available_columns].copy()
    
    # Convert LOCAL_DATE to datetime
    if 'LOCAL_DATE' in cleaned_df.columns:
        cleaned_df['LOCAL_DATE'] = pd.to_datetime(cleaned_df['LOCAL_DATE'], errors='coerce')
    
    print(f"✅ Cleaned weather data: {len(cleaned_df)} records, {len(cleaned_df.columns)} columns")
    return cleaned_df

def save_weather_data(weather_df, filename='toronto_weather_data.csv'):
    """
    Save weather data to CSV file in the Data directory.
    
    Args:
        weather_df (pandas.DataFrame): Weather data to save
        filename (str): Output filename
    
    Returns:
        str: Path to saved file, or None if failed
    """
    if weather_df is None or weather_df.empty:
        print("❌ No weather data to save")
        return None
    
    # Save in the main Data directory
    output_path = os.path.join('Data', filename)
    
    try:
        weather_df.to_csv(output_path, index=False)
        print(f"✅ Weather data saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error saving weather data: {e}")
        return None

def main():
    """
    Main function to fetch, clean, and save weather data for Toronto/Ontario.
    """
    print("🌤️ Starting Weather Data Collection for Toronto Hotel Analysis")
    
    # Configuration
    years_to_fetch = [2025]  # Can be extended to include [2024, 2025]
    province_code = 'ON'
    
    # Step 1: Fetch weather data
    raw_weather_data = fetch_weather_data_for_years(years_to_fetch, province_code)
    
    if raw_weather_data is not None:
        # Step 2: Clean weather data
        cleaned_weather_data = clean_weather_data(raw_weather_data)
        
        if cleaned_weather_data is not None:
            # Step 3: Save cleaned data
            save_path = save_weather_data(cleaned_weather_data, 'toronto_weather_2025.csv')
            
            if save_path:
                print(f"\n🎉 Weather data collection completed successfully!")
                print(f"📊 Total records: {len(cleaned_weather_data)}")
                print(f"📅 Date range: {cleaned_weather_data['LOCAL_DATE'].min()} to {cleaned_weather_data['LOCAL_DATE'].max()}")
            else:
                print("\n❌ Failed to save weather data")
        else:
            print("\n❌ Failed to clean weather data")
    else:
        print("\n❌ Failed to fetch weather data")

if __name__ == "__main__":
    main()