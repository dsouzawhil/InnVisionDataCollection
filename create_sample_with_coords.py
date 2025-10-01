"""
Create a sample dataset with real coordinates to demonstrate spatial analysis.
"""

import pandas as pd
from geopy.geocoders import Nominatim
import time

def create_sample_dataset():
    """Create a small sample with geocoded hotels for demonstration."""
    print("🎯 Creating sample dataset with real coordinates...")
    
    # Load original data
    hotels_df = pd.read_csv('Data/toronto_hotels_cleaned_transformed.csv')
    events_df = pd.read_csv('Data/toronto_events_cleaned.csv')
    
    # Take first 10 hotels for sample
    sample_hotels = hotels_df.head(10).copy()
    
    # Geocode these hotels quickly
    geocoder = Nominatim(user_agent='sample_creator_v1.0', timeout=5)
    
    sample_hotels['hotel_latitude'] = None
    sample_hotels['hotel_longitude'] = None
    sample_hotels['geocoding_status'] = None
    
    print(f"📍 Geocoding {len(sample_hotels)} sample hotels...")
    
    success_count = 0
    for idx, hotel in sample_hotels.iterrows():
        address = str(hotel['Address']) + ', Toronto, Ontario, Canada'
        
        try:
            location = geocoder.geocode(address)
            if location:
                sample_hotels.loc[idx, 'hotel_latitude'] = location.latitude
                sample_hotels.loc[idx, 'hotel_longitude'] = location.longitude
                sample_hotels.loc[idx, 'geocoding_status'] = 'Success'
                success_count += 1
                print(f"   ✅ {hotel['Hotel name'][:30]}: ({location.latitude:.4f}, {location.longitude:.4f})")
            else:
                sample_hotels.loc[idx, 'geocoding_status'] = 'No results'
                print(f"   ❌ {hotel['Hotel name'][:30]}: No location found")
        
        except Exception as e:
            sample_hotels.loc[idx, 'geocoding_status'] = f'Error: {str(e)}'
            print(f"   ❌ {hotel['Hotel name'][:30]}: Error - {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    print(f"\n✅ Geocoded {success_count}/{len(sample_hotels)} hotels successfully")
    
    # Save sample datasets
    sample_hotels.to_csv('Data/sample_hotels_with_coordinates.csv', index=False)
    
    # Take a subset of events for the sample (events during our sample date range)
    sample_events = events_df[
        pd.to_datetime(events_df['Date']).between('2025-10-10', '2025-10-20')
    ].copy()
    
    sample_events.to_csv('Data/sample_events.csv', index=False)
    
    print(f"💾 Saved sample datasets:")
    print(f"   🏨 {len(sample_hotels)} hotels -> Data/sample_hotels_with_coordinates.csv")
    print(f"   🎭 {len(sample_events)} events -> Data/sample_events.csv")
    
    return sample_hotels, sample_events

if __name__ == "__main__":
    create_sample_dataset()