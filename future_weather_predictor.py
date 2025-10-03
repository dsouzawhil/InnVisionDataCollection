"""
Future Weather Prediction for Hotel Analysis
===========================================

This module provides multiple approaches to get weather data for future dates
in hotel price prediction models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os

def get_historical_weather_patterns(years_back=5):
    """
    Analyze historical weather patterns to predict future weather.
    Uses existing weather data to create seasonal averages.
    """
    print("📊 Analyzing historical weather patterns...")
    
    # Check if we have historical weather data
    weather_file = 'Data/toronto_weather_cleaned.csv'
    if not os.path.exists(weather_file):
        print(f"❌ No historical weather data found at {weather_file}")
        return None
    
    # Load historical weather data
    weather_df = pd.read_csv(weather_file)
    weather_df['LOCAL_DATE'] = pd.to_datetime(weather_df['LOCAL_DATE'])
    
    # Extract month and day for seasonal patterns
    weather_df['month'] = weather_df['LOCAL_DATE'].dt.month
    weather_df['day_of_year'] = weather_df['LOCAL_DATE'].dt.dayofyear
    
    # Create seasonal averages
    seasonal_stats = weather_df.groupby(['month']).agg({
        'MEAN_TEMPERATURE': ['mean', 'std'],
        'MIN_TEMPERATURE': ['mean', 'std'],
        'MAX_TEMPERATURE': ['mean', 'std'],
        'TOTAL_PRECIPITATION': ['mean', 'std'],
        'TOTAL_SNOW': ['mean', 'std'],
        'SNOW_ON_GROUND': ['mean', 'std'],
        'SPEED_MAX_GUST': ['mean', 'std'],
        'MAX_REL_HUMIDITY': ['mean', 'std']
    }).round(2)
    
    # Flatten column names
    seasonal_stats.columns = [f'{col[0]}_{col[1]}' for col in seasonal_stats.columns]
    
    print(f"✅ Created seasonal weather patterns from {len(weather_df)} historical records")
    return seasonal_stats

def predict_weather_from_patterns(target_dates, seasonal_stats, add_variation=True):
    """
    Predict weather for future dates based on historical seasonal patterns.
    """
    if seasonal_stats is None:
        return None
    
    print(f"🔮 Predicting weather for {len(target_dates)} future dates...")
    
    predicted_weather = []
    
    for date in target_dates:
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        month = date.month
        
        if month in seasonal_stats.index:
            # Get seasonal averages for this month
            month_stats = seasonal_stats.loc[month]
            
            weather_record = {
                'LOCAL_DATE': date,
                'MEAN_TEMPERATURE': month_stats['MEAN_TEMPERATURE_mean'],
                'MIN_TEMPERATURE': month_stats['MIN_TEMPERATURE_mean'],
                'MAX_TEMPERATURE': month_stats['MAX_TEMPERATURE_mean'],
                'TOTAL_PRECIPITATION': month_stats['TOTAL_PRECIPITATION_mean'],
                'TOTAL_SNOW': month_stats['TOTAL_SNOW_mean'],
                'SNOW_ON_GROUND': month_stats['SNOW_ON_GROUND_mean'],
                'SPEED_MAX_GUST': month_stats['SPEED_MAX_GUST_mean'],
                'MAX_REL_HUMIDITY': month_stats['MAX_REL_HUMIDITY_mean']
            }
            
            # Add natural variation if requested
            if add_variation:
                for col in ['MEAN_TEMPERATURE', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE']:
                    if f'{col}_std' in month_stats:
                        std_dev = month_stats[f'{col}_std']
                        # Add random variation within 0.5 standard deviations
                        variation = np.random.normal(0, std_dev * 0.5)
                        weather_record[col] += variation
            
            predicted_weather.append(weather_record)
    
    if predicted_weather:
        predicted_df = pd.DataFrame(predicted_weather)
        print(f"✅ Generated weather predictions for {len(predicted_df)} dates")
        return predicted_df
    else:
        print("❌ No weather predictions generated")
        return None

def get_weather_api_data(api_key=None, location="Toronto"):
    """
    Get weather forecast from weather API services.
    Note: Most free APIs provide 7-14 day forecasts only.
    """
    if not api_key:
        print("⚠️ No API key provided for weather service")
        print("💡 Available free weather APIs:")
        print("   • OpenWeatherMap (free tier: 5-day forecast)")
        print("   • WeatherAPI (free tier: 10-day forecast)")
        print("   • AccuWeather (free tier: 5-day forecast)")
        return None
    
    # Example implementation for OpenWeatherMap
    try:
        # 5-day forecast endpoint
        url = f"http://api.openweathermap.org/data/2.5/forecast"
        params = {
            'q': location,
            'appid': api_key,
            'units': 'metric',
            'cnt': 40  # 5 days * 8 (3-hour intervals)
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Process the forecast data
        forecasts = []
        for item in data['list']:
            forecast = {
                'LOCAL_DATE': pd.to_datetime(item['dt'], unit='s'),
                'MEAN_TEMPERATURE': item['main']['temp'],
                'MIN_TEMPERATURE': item['main']['temp_min'],
                'MAX_TEMPERATURE': item['main']['temp_max'],
                'TOTAL_PRECIPITATION': item.get('rain', {}).get('3h', 0) + item.get('snow', {}).get('3h', 0),
                'MAX_REL_HUMIDITY': item['main']['humidity'],
                'SPEED_MAX_GUST': item['wind'].get('gust', item['wind']['speed'])
            }
            forecasts.append(forecast)
        
        forecast_df = pd.DataFrame(forecasts)
        # Group by date to get daily averages
        daily_forecast = forecast_df.groupby(forecast_df['LOCAL_DATE'].dt.date).agg({
            'MEAN_TEMPERATURE': 'mean',
            'MIN_TEMPERATURE': 'min',
            'MAX_TEMPERATURE': 'max',
            'TOTAL_PRECIPITATION': 'sum',
            'MAX_REL_HUMIDITY': 'max',
            'SPEED_MAX_GUST': 'max'
        }).round(2)
        
        print(f"✅ Retrieved {len(daily_forecast)} days of weather forecast from API")
        return daily_forecast
        
    except Exception as e:
        print(f"❌ Error fetching weather data from API: {e}")
        return None

def create_simple_weather_estimates():
    """
    Create simple weather estimates based on Toronto's climate patterns.
    This is the most basic approach when no historical data is available.
    """
    print("🌤️ Creating simple weather estimates based on Toronto climate...")
    
    # Toronto typical weather by month (October focus)
    toronto_climate = {
        'october': {
            'MEAN_TEMPERATURE': 12.0,  # °C
            'MIN_TEMPERATURE': 7.0,
            'MAX_TEMPERATURE': 17.0,
            'TOTAL_PRECIPITATION': 2.5,  # mm per day average
            'TOTAL_SNOW': 0.0,  # Rarely snows in October
            'SNOW_ON_GROUND': 0.0,
            'SPEED_MAX_GUST': 15.0,  # km/h
            'MAX_REL_HUMIDITY': 75.0  # %
        }
    }
    
    return toronto_climate['october']

def update_unified_dataset_with_weather(method='historical_patterns', api_key=None):
    """
    Update the unified hotel dataset with weather predictions.
    """
    print("🔄 Updating unified dataset with weather predictions...")
    
    # Load the unified dataset
    unified_file = 'Data/toronto_unified_hotel_analysis.csv'
    if not os.path.exists(unified_file):
        print(f"❌ Unified dataset not found: {unified_file}")
        return
    
    df = pd.read_csv(unified_file)
    df['Check-in Date'] = pd.to_datetime(df['Check-in Date'])
    
    # Get unique check-in dates
    unique_dates = df['Check-in Date'].dropna().unique()
    print(f"📅 Found {len(unique_dates)} unique check-in dates")
    
    weather_data = None
    
    if method == 'historical_patterns':
        # Use historical weather patterns
        seasonal_stats = get_historical_weather_patterns()
        if seasonal_stats is not None:
            weather_data = predict_weather_from_patterns(unique_dates, seasonal_stats)
    
    elif method == 'api':
        # Use weather API (limited to short-term forecasts)
        weather_data = get_weather_api_data(api_key)
    
    elif method == 'simple_estimates':
        # Use simple climate estimates
        climate_data = create_simple_weather_estimates()
        weather_records = []
        
        for date in unique_dates:
            record = climate_data.copy()
            record['LOCAL_DATE'] = date
            # Add small random variations
            for temp_col in ['MEAN_TEMPERATURE', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE']:
                record[temp_col] += np.random.normal(0, 2)  # ±2°C variation
            weather_records.append(record)
        
        weather_data = pd.DataFrame(weather_records)
        print(f"✅ Created simple weather estimates for {len(weather_data)} dates")
    
    if weather_data is not None:
        # Merge weather data with hotel data
        df_updated = df.merge(
            weather_data,
            left_on='Check-in Date',
            right_on='LOCAL_DATE',
            how='left'
        )
        
        # Drop the extra date column
        if 'LOCAL_DATE' in df_updated.columns:
            df_updated = df_updated.drop('LOCAL_DATE', axis=1)
        
        # Fill any remaining missing weather data with estimates
        weather_cols = [
            'MEAN_TEMPERATURE', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE',
            'TOTAL_PRECIPITATION', 'TOTAL_SNOW', 'SNOW_ON_GROUND',
            'SPEED_MAX_GUST', 'MAX_REL_HUMIDITY'
        ]
        
        for col in weather_cols:
            if col in df_updated.columns:
                missing_count = df_updated[col].isnull().sum()
                if missing_count > 0:
                    # Fill with simple estimates
                    if 'october' in locals():
                        fill_value = create_simple_weather_estimates().get(col, 0)
                        df_updated[col].fillna(fill_value, inplace=True)
                        print(f"   Filled {missing_count} missing values in {col}")
        
        # Save updated dataset
        output_file = 'Data/toronto_unified_hotel_analysis_with_weather.csv'
        df_updated.to_csv(output_file, index=False)
        
        print(f"✅ Updated dataset saved to: {output_file}")
        print(f"📊 Records: {len(df_updated)}, Columns: {len(df_updated.columns)}")
        
        # Show weather data coverage
        weather_coverage = {}
        for col in weather_cols:
            if col in df_updated.columns:
                non_null = df_updated[col].notna().sum()
                coverage = (non_null / len(df_updated)) * 100
                weather_coverage[col] = coverage
        
        print(f"\n🌤️ Weather Data Coverage:")
        for col, coverage in weather_coverage.items():
            print(f"   {col}: {coverage:.1f}%")
        
        return df_updated
    else:
        print("❌ No weather data could be generated")
        return None

def setup_weather_update_scheduler():
    """
    Create a script to automatically update weather data as dates pass.
    """
    scheduler_script = '''
import pandas as pd
from datetime import datetime, date
import requests

def update_actual_weather():
    """Update weather data with actual observations as dates pass."""
    print(f"🔄 Checking for weather updates on {datetime.now().strftime('%Y-%m-%d')}")
    
    # Load the dataset
    df = pd.read_csv('Data/toronto_unified_hotel_analysis_with_weather.csv')
    df['Check-in Date'] = pd.to_datetime(df['Check-in Date'])
    
    # Find dates that have now passed
    today = pd.Timestamp(date.today())
    past_dates = df[df['Check-in Date'] <= today]['Check-in Date'].unique()
    
    if len(past_dates) > 0:
        print(f"📅 Found {len(past_dates)} dates to update with actual weather")
        
        # Here you would fetch actual weather data
        # For now, we'll just flag these records
        df['weather_updated'] = df['Check-in Date'] <= today
        
        # Save updated dataset
        df.to_csv('Data/toronto_unified_hotel_analysis_with_weather.csv', index=False)
        print(f"✅ Marked {len(past_dates)} dates for weather update")
    else:
        print("ℹ️ No dates have passed yet - no weather updates needed")

if __name__ == "__main__":
    update_actual_weather()
'''
    
    with open('weather_update_scheduler.py', 'w') as f:
        f.write(scheduler_script)
    
    print("✅ Created weather_update_scheduler.py")
    print("💡 Run this script daily to update past dates with actual weather data")

def main():
    """Main function to demonstrate weather prediction approaches."""
    print("🌤️ FUTURE WEATHER PREDICTION FOR HOTEL ANALYSIS")
    print("=" * 55)
    
    print("\n🎯 Available Approaches:")
    print("1. Historical Patterns (Recommended)")
    print("2. Weather API (Short-term only)")
    print("3. Simple Climate Estimates (Fallback)")
    
    # Try historical patterns first
    print(f"\n{'='*50}")
    print("APPROACH 1: Historical Weather Patterns")
    print("="*50)
    
    result = update_unified_dataset_with_weather(method='historical_patterns')
    
    if result is None:
        print(f"\n{'='*50}")
        print("APPROACH 3: Simple Climate Estimates (Fallback)")
        print("="*50)
        result = update_unified_dataset_with_weather(method='simple_estimates')
    
    # Setup future weather update scheduler
    setup_weather_update_scheduler()
    
    print(f"\n🎉 Weather prediction setup complete!")
    print(f"💡 For API-based forecasts, get a free API key from:")
    print(f"   • OpenWeatherMap: https://openweathermap.org/api")
    print(f"   • WeatherAPI: https://www.weatherapi.com/")

if __name__ == "__main__":
    main()