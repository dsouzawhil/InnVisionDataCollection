"""
Geocode Hotel Addresses to Get Latitude and Longitude
====================================================

This script adds latitude and longitude coordinates to hotel addresses.
"""

import pandas as pd
import time
import requests
from urllib.parse import quote
import numpy as np

def geocode_with_nominatim(address, city="Toronto", country="Canada"):
    """
    Geocode address using Nominatim (OpenStreetMap) - Free service.
    """
    if pd.isna(address) or str(address).strip() == '':
        return None, None
    
    # Clean and format address
    full_address = f"{address}, {city}, {country}"
    encoded_address = quote(full_address)
    
    # Nominatim API endpoint
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={encoded_address}&limit=1"
    
    try:
        # Add delay to respect rate limits (1 request per second)
        time.sleep(1.1)
        
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'TorontoHotelAnalysis/1.0 (hotel.analysis@example.com)'
        })
        response.raise_for_status()
        
        data = response.json()
        
        if data and len(data) > 0:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon
        else:
            return None, None
            
    except Exception as e:
        print(f"   ⚠️ Error geocoding '{address}': {e}")
        return None, None

def geocode_simple_toronto(district):
    """
    Get approximate coordinates for Toronto districts when address is just 'Toronto'.
    """
    # Approximate coordinates for Toronto districts/areas
    district_coords = {
        'Entertainment District': (43.6426, -79.3871),
        'Financial District': (43.6489, -79.3817),
        'Fashion District': (43.6505, -79.3934),
        'Downtown Toronto': (43.6532, -79.3832),
        'Bloor-Yorkville': (43.6706, -79.3951),
        'The Annex': (43.6677, -79.4103),
        'Queen West': (43.6430, -79.4016),
        'King Street West': (43.6441, -79.4005),
        'Liberty Village': (43.6362, -79.4194),
        'Distillery District': (43.6503, -79.3592),
        'St. Lawrence': (43.6485, -79.3713),
        'Old Town': (43.6503, -79.3713),
        'Corktown': (43.6525, -79.3635),
        'Harbourfront': (43.6385, -79.3762),
        'CityPlace': (43.6394, -79.3862),
        'Church-Wellesley': (43.6641, -79.3832),
        'Yonge - Dundas': (43.6555, -79.3807),
        'The Village': (43.6641, -79.3832),
        'Kensington Market': (43.6547, -79.4025),
        'Chinatown': (43.6530, -79.3985),
        'Little Italy': (43.6547, -79.4183),
        'Trinity Bellwoods': (43.6478, -79.4183),
        'Parkdale': (43.6367, -79.4331),
        'High Park': (43.6465, -79.4637),
        'Junction': (43.6504, -79.4748),
        'Leslieville': (43.6617, -79.3382),
        'Riverdale': (43.6617, -79.3516),
        'Beaches': (43.6677, -79.2931),
        'Cabbagetown': (43.6617, -79.3650),
        'Regent Park': (43.6568, -79.3650),
        
        # Outer areas
        'Etobicoke': (43.6205, -79.5132),
        'Scarborough': (43.7731, -79.2578),
        'North York': (43.7615, -79.4111),
        'York': (43.6896, -79.4411),
        'East York': (43.6896, -79.3254),
        
        # Airport/Mississauga
        'Airport': (43.6777, -79.6248),
        'Northeast Mississauga': (43.5890, -79.6441),
        'Mississauga': (43.5890, -79.6441),
        
        # Additional areas
        'Yorkdale': (43.7252, -79.4522),
        'Danforth': (43.6778, -79.3489),
        'College': (43.6577, -79.4025)
    }
    
    # Try exact match first
    if district in district_coords:
        return district_coords[district]
    
    # Try partial matches
    district_lower = district.lower() if district else ''
    for key, coords in district_coords.items():
        if district_lower in key.lower() or key.lower() in district_lower:
            return coords
    
    # Default to downtown Toronto
    return (43.6532, -79.3832)

def add_coordinates_to_dataset(df, method='mixed'):
    """
    Add latitude and longitude coordinates to the hotel dataset.
    
    Methods:
    - 'api': Use geocoding API for all addresses
    - 'district': Use district-based coordinates only
    - 'mixed': Use API for detailed addresses, district coords for 'Toronto' only
    """
    print("🗺️ ADDING COORDINATES TO HOTEL DATASET")
    print("=" * 42)
    
    if 'Address' not in df.columns:
        print("❌ No Address column found")
        return df
    
    # Add coordinate columns if they don't exist
    if 'Latitude' not in df.columns:
        df['Latitude'] = np.nan
    if 'Longitude' not in df.columns:
        df['Longitude'] = np.nan
    
    # Count existing coordinates
    existing_coords = df[['Latitude', 'Longitude']].notna().all(axis=1).sum()
    total_records = len(df)
    
    print(f"📊 Initial state:")
    print(f"   Total records: {total_records:,}")
    print(f"   Existing coordinates: {existing_coords:,} ({existing_coords/total_records*100:.1f}%)")
    print(f"   Missing coordinates: {total_records - existing_coords:,}")
    
    if method == 'mixed':
        print(f"\n🔄 Using mixed approach:")
        print(f"   • API geocoding for detailed addresses")
        print(f"   • District coordinates for 'Toronto' addresses")
        
        # Identify records needing geocoding
        missing_coords = df[['Latitude', 'Longitude']].isna().any(axis=1)
        detailed_addresses = missing_coords & (df['Address'] != 'Toronto') & df['Address'].notna()
        toronto_only = missing_coords & (df['Address'] == 'Toronto')
        
        print(f"\n📍 Address breakdown:")
        print(f"   Detailed addresses to geocode: {detailed_addresses.sum():,}")
        print(f"   'Toronto' only addresses: {toronto_only.sum():,}")
        
        # Geocode detailed addresses with API
        if detailed_addresses.sum() > 0:
            print(f"\n🌐 Geocoding {detailed_addresses.sum()} detailed addresses...")
            
            for idx, row in df[detailed_addresses].iterrows():
                if idx % 50 == 0:  # Progress update every 50 records
                    print(f"   Progress: {idx}/{len(df)} records")
                
                address = row['Address']
                lat, lon = geocode_with_nominatim(address)
                
                if lat is not None and lon is not None:
                    df.loc[idx, 'Latitude'] = lat
                    df.loc[idx, 'Longitude'] = lon
        
        # Use district coordinates for 'Toronto' addresses
        if toronto_only.sum() > 0:
            print(f"\n📍 Using district coordinates for {toronto_only.sum()} 'Toronto' addresses...")
            
            for idx, row in df[toronto_only].iterrows():
                district = row.get('district')
                if pd.notna(district):
                    lat, lon = geocode_simple_toronto(district)
                    df.loc[idx, 'Latitude'] = lat
                    df.loc[idx, 'Longitude'] = lon
                else:
                    # Default to downtown Toronto if no district
                    df.loc[idx, 'Latitude'] = 43.6532
                    df.loc[idx, 'Longitude'] = -79.3832
    
    elif method == 'district':
        print(f"\n📍 Using district-based coordinates only...")
        
        missing_coords = df[['Latitude', 'Longitude']].isna().any(axis=1)
        
        for idx, row in df[missing_coords].iterrows():
            district = row.get('district')
            lat, lon = geocode_simple_toronto(district)
            df.loc[idx, 'Latitude'] = lat
            df.loc[idx, 'Longitude'] = lon
    
    elif method == 'api':
        print(f"\n🌐 Using API geocoding for all addresses...")
        
        missing_coords = df[['Latitude', 'Longitude']].isna().any(axis=1)
        
        for idx, row in df[missing_coords].iterrows():
            if idx % 50 == 0:
                print(f"   Progress: {idx}/{len(df)} records")
            
            address = row['Address']
            lat, lon = geocode_with_nominatim(address)
            
            if lat is not None and lon is not None:
                df.loc[idx, 'Latitude'] = lat
                df.loc[idx, 'Longitude'] = lon
    
    # Final statistics
    final_coords = df[['Latitude', 'Longitude']].notna().all(axis=1).sum()
    coverage = final_coords / total_records * 100
    
    print(f"\n✅ Geocoding complete:")
    print(f"   Final coordinates: {final_coords:,} ({coverage:.1f}%)")
    print(f"   Added coordinates: {final_coords - existing_coords:,}")
    print(f"   Still missing: {total_records - final_coords:,}")
    
    return df

def main():
    """Main function to test geocoding."""
    print("🧪 TESTING HOTEL ADDRESS GEOCODING")
    print("=" * 38)
    
    # Load dataset with filled districts
    df = pd.read_csv('Data/toronto_unified_hotel_analysis_districts_filled.csv')
    
    print(f"📂 Loaded dataset: {len(df):,} rows")
    
    # Add coordinates using mixed approach
    df_with_coords = add_coordinates_to_dataset(df.copy(), method='mixed')
    
    # Show some examples
    print(f"\n📋 Sample coordinates:")
    sample = df_with_coords[['Hotel name', 'Address', 'district', 'Latitude', 'Longitude']].head(10)
    
    for idx, row in sample.iterrows():
        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            print(f"   {row['Hotel name'][:40]}...")
            print(f"   📍 {row['Latitude']:.4f}, {row['Longitude']:.4f}")
            print(f"   📌 {row['district']} | {row['Address']}")
            print()
    
    # Save dataset with coordinates
    output_file = 'Data/toronto_unified_hotel_analysis_with_coordinates.csv'
    df_with_coords.to_csv(output_file, index=False)
    print(f"✅ Dataset with coordinates saved to: {output_file}")
    
    return df_with_coords

if __name__ == "__main__":
    main()