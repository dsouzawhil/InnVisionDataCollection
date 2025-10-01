"""
Hotel Address Geocoding Script

This script adds latitude and longitude coordinates to hotel data by geocoding
their addresses using the Nominatim API (free OpenStreetMap geocoding service).
"""

import pandas as pd
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import warnings
warnings.filterwarnings('ignore')

def setup_geocoder():
    """Initialize the geocoder with proper user agent."""
    return Nominatim(
        user_agent="toronto_hotel_analysis_v1.0",
        timeout=10
    )

def clean_address_for_geocoding(address):
    """Clean and format address for better geocoding results."""
    if pd.isna(address) or address == '':
        return None
    
    # Clean the address
    address = str(address).strip()
    
    # Remove common booking.com artifacts
    address = address.replace('(Entertainment District)', '')
    address = address.replace('(Downtown Toronto)', '')
    address = address.replace('(Kensington Market)', '')
    address = address.replace('(The Village)', '')
    address = address.replace('(Yonge - Dundas)', '')
    
    # Add Toronto and Ontario to improve geocoding
    if 'Toronto' not in address:
        address += ', Toronto'
    if 'Ontario' not in address and 'ON' not in address:
        address += ', Ontario, Canada'
    
    return address.strip()

def geocode_address_with_retry(geocoder, address, max_retries=3):
    """Geocode an address with retry logic."""
    if not address:
        return None, None, "Empty address"
    
    for attempt in range(max_retries):
        try:
            location = geocoder.geocode(address)
            if location:
                return location.latitude, location.longitude, "Success"
            else:
                return None, None, f"No results found for: {address}"
        
        except GeocoderTimedOut:
            if attempt < max_retries - 1:
                print(f"   ⏰ Timeout, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2)
            else:
                return None, None, "Geocoding timeout"
        
        except GeocoderServiceError as e:
            return None, None, f"Geocoding service error: {e}"
        
        except Exception as e:
            return None, None, f"Unexpected error: {e}"
    
    return None, None, "Max retries exceeded"

def geocode_hotels(hotels_df):
    """Add latitude and longitude coordinates to hotel dataframe."""
    print("🗺️ Starting hotel address geocoding...")
    print(f"   📊 Total hotels to geocode: {len(hotels_df)}")
    
    # Initialize geocoder
    geocoder = setup_geocoder()
    
    # Add new columns for coordinates
    hotels_df = hotels_df.copy()
    hotels_df['hotel_latitude'] = None
    hotels_df['hotel_longitude'] = None
    hotels_df['geocoding_status'] = None
    hotels_df['cleaned_address'] = None
    
    successful_geocodes = 0
    failed_geocodes = 0
    
    # Process each hotel
    for idx, hotel in hotels_df.iterrows():
        if idx % 100 == 0:
            print(f"   🔄 Processing hotel {idx + 1}/{len(hotels_df)} ({successful_geocodes} successful)")
        
        # Clean the address
        original_address = hotel.get('Address', '')
        cleaned_address = clean_address_for_geocoding(original_address)
        hotels_df.loc[idx, 'cleaned_address'] = cleaned_address
        
        # Geocode the address
        lat, lon, status = geocode_address_with_retry(geocoder, cleaned_address)
        
        # Store results
        hotels_df.loc[idx, 'hotel_latitude'] = lat
        hotels_df.loc[idx, 'hotel_longitude'] = lon
        hotels_df.loc[idx, 'geocoding_status'] = status
        
        if lat is not None and lon is not None:
            successful_geocodes += 1
        else:
            failed_geocodes += 1
            if idx < 5:  # Show first few failures for debugging
                print(f"   ❌ Failed: {original_address} -> {status}")
        
        # Rate limiting - be respectful to free service
        time.sleep(0.5)  # 0.5 seconds between requests
    
    print(f"\n✅ Geocoding completed!")
    print(f"   🎯 Successful: {successful_geocodes}/{len(hotels_df)} ({successful_geocodes/len(hotels_df)*100:.1f}%)")
    print(f"   ❌ Failed: {failed_geocodes}/{len(hotels_df)} ({failed_geocodes/len(hotels_df)*100:.1f}%)")
    
    return hotels_df

def validate_coordinates(hotels_df):
    """Validate that coordinates are reasonable for Toronto area."""
    print("\n🔍 Validating coordinates...")
    
    # Toronto bounding box (approximate)
    toronto_bounds = {
        'lat_min': 43.4, 'lat_max': 44.0,
        'lon_min': -79.8, 'lon_max': -79.0
    }
    
    valid_coords = hotels_df.dropna(subset=['hotel_latitude', 'hotel_longitude'])
    
    if len(valid_coords) == 0:
        print("   ❌ No valid coordinates found!")
        return
    
    # Check if coordinates are in reasonable range
    in_bounds = valid_coords[
        (valid_coords['hotel_latitude'] >= toronto_bounds['lat_min']) &
        (valid_coords['hotel_latitude'] <= toronto_bounds['lat_max']) &
        (valid_coords['hotel_longitude'] >= toronto_bounds['lon_min']) &
        (valid_coords['hotel_longitude'] <= toronto_bounds['lon_max'])
    ]
    
    print(f"   📍 Coordinates in Toronto area: {len(in_bounds)}/{len(valid_coords)} ({len(in_bounds)/len(valid_coords)*100:.1f}%)")
    
    # Show coordinate statistics
    print(f"   📊 Latitude range: {valid_coords['hotel_latitude'].min():.4f} to {valid_coords['hotel_latitude'].max():.4f}")
    print(f"   📊 Longitude range: {valid_coords['hotel_longitude'].min():.4f} to {valid_coords['hotel_longitude'].max():.4f}")
    
    # Show some examples
    print(f"\n📍 Sample geocoded hotels:")
    sample_hotels = valid_coords.head(3)
    for _, hotel in sample_hotels.iterrows():
        print(f"   {hotel['Hotel name'][:50]}: ({hotel['hotel_latitude']:.4f}, {hotel['hotel_longitude']:.4f})")

def save_geocoded_hotels(hotels_df, filename='toronto_hotels_with_coordinates.csv'):
    """Save the geocoded hotel data."""
    output_path = f"Data/{filename}"
    
    try:
        hotels_df.to_csv(output_path, index=False)
        print(f"\n✅ Geocoded hotel data saved to: {output_path}")
        print(f"   📊 Records: {len(hotels_df)}")
        print(f"   📊 With coordinates: {hotels_df['hotel_latitude'].notna().sum()}")
        
        return output_path
    except Exception as e:
        print(f"❌ Error saving geocoded data: {e}")
        return None

def main():
    """Main function to geocode hotel addresses."""
    print("🗺️ HOTEL ADDRESS GEOCODING PIPELINE")
    print("=" * 50)
    
    # Load hotel data
    print("\n1️⃣ Loading hotel data...")
    try:
        hotels_df = pd.read_csv('Data/toronto_hotels_cleaned_transformed.csv')
        print(f"✅ Loaded {len(hotels_df)} hotel records")
    except Exception as e:
        print(f"❌ Error loading hotel data: {e}")
        return
    
    # Check if already geocoded
    if 'hotel_latitude' in hotels_df.columns and 'hotel_longitude' in hotels_df.columns:
        existing_coords = hotels_df['hotel_latitude'].notna().sum()
        print(f"⚠️ Found existing coordinates for {existing_coords} hotels")
        response = input("Do you want to re-geocode all addresses? (y/n): ").lower()
        if response != 'y':
            print("Skipping geocoding...")
            return
    
    # Geocode addresses
    print("\n2️⃣ Geocoding hotel addresses...")
    geocoded_df = geocode_hotels(hotels_df)
    
    # Validate results
    validate_coordinates(geocoded_df)
    
    # Save results
    print("\n3️⃣ Saving geocoded data...")
    save_path = save_geocoded_hotels(geocoded_df)
    
    if save_path:
        print(f"\n🎉 Hotel geocoding completed successfully!")
        print(f"📁 Geocoded data available at: {save_path}")
        print(f"💡 Ready for spatial analysis with real hotel coordinates!")
    else:
        print(f"\n❌ Failed to save geocoded data")

if __name__ == "__main__":
    main()