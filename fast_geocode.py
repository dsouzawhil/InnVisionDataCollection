"""
Fast Hotel Geocoding Script

This script geocodes hotels in small batches and saves progress incrementally.
"""

import pandas as pd
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import warnings
warnings.filterwarnings('ignore')

def geocode_batch(hotels_df, start_idx=0, batch_size=50):
    """Geocode a batch of hotels and save progress."""
    print(f"🗺️ Geocoding batch: hotels {start_idx} to {start_idx + batch_size}")
    
    geocoder = Nominatim(user_agent="toronto_hotel_fast_v1.0", timeout=10)
    
    # Initialize columns if they don't exist
    if 'hotel_latitude' not in hotels_df.columns:
        hotels_df['hotel_latitude'] = None
        hotels_df['hotel_longitude'] = None
        hotels_df['geocoding_status'] = None
    
    success_count = 0
    end_idx = min(start_idx + batch_size, len(hotels_df))
    
    for idx in range(start_idx, end_idx):
        hotel = hotels_df.iloc[idx]
        
        # Skip if already geocoded
        if pd.notna(hotels_df.iloc[idx]['hotel_latitude']):
            success_count += 1
            continue
        
        # Clean address
        address = str(hotel['Address']) + ', Toronto, Ontario, Canada'
        
        try:
            location = geocoder.geocode(address)
            if location:
                hotels_df.loc[hotels_df.index[idx], 'hotel_latitude'] = location.latitude
                hotels_df.loc[hotels_df.index[idx], 'hotel_longitude'] = location.longitude
                hotels_df.loc[hotels_df.index[idx], 'geocoding_status'] = 'Success'
                success_count += 1
                print(f"   ✅ {idx}: {hotel['Hotel name'][:30]} -> ({location.latitude:.4f}, {location.longitude:.4f})")
            else:
                hotels_df.loc[hotels_df.index[idx], 'geocoding_status'] = 'No results'
                print(f"   ❌ {idx}: {hotel['Hotel name'][:30]} -> No results")
        
        except Exception as e:
            hotels_df.loc[hotels_df.index[idx], 'geocoding_status'] = f'Error: {str(e)[:50]}'
            print(f"   ❌ {idx}: {hotel['Hotel name'][:30]} -> Error: {e}")
        
        # Small delay between requests
        time.sleep(0.3)
    
    print(f"   📊 Batch complete: {success_count}/{end_idx - start_idx} successful")
    return success_count

def main():
    """Fast geocoding with progress saving."""
    print("🚀 FAST HOTEL GEOCODING")
    print("=" * 30)
    
    # Load hotel data
    hotels_df = pd.read_csv('Data/toronto_hotels_cleaned_transformed.csv')
    print(f"📊 Total hotels: {len(hotels_df)}")
    
    # Process first 100 hotels as a test
    batch_size = 100
    print(f"🧪 Processing first {batch_size} hotels as test...")
    
    geocode_batch(hotels_df, start_idx=0, batch_size=batch_size)
    
    # Save the geocoded batch
    output_path = 'Data/toronto_hotels_with_coordinates.csv'
    hotels_df.to_csv(output_path, index=False)
    
    # Show results
    geocoded_count = hotels_df['hotel_latitude'].notna().sum()
    print(f"\n✅ Geocoding test complete!")
    print(f"📍 Successfully geocoded: {geocoded_count}/{len(hotels_df)} hotels")
    print(f"💾 Saved to: {output_path}")
    
    if geocoded_count > 0:
        print(f"\n📊 Sample geocoded hotels:")
        sample = hotels_df[hotels_df['hotel_latitude'].notna()].head(3)
        for _, hotel in sample.iterrows():
            print(f"   {hotel['Hotel name'][:40]}: ({hotel['hotel_latitude']:.4f}, {hotel['hotel_longitude']:.4f})")

if __name__ == "__main__":
    main()