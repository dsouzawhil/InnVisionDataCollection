
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
