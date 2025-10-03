"""
Quick Geocoding - Primarily District-Based
==========================================

Fast geocoding using district coordinates for most addresses.
"""

import pandas as pd
import numpy as np

def get_district_coordinates():
    """Get coordinates for Toronto districts."""
    return {
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
        'The Harbourfront': (43.6385, -79.3762),
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
        'The Beaches': (43.6677, -79.2931),
        'Cabbagetown': (43.6617, -79.3650),
        'Regent Park': (43.6568, -79.3650),
        'West Queen West': (43.6430, -79.4116),
        'Yonge Street': (43.6555, -79.3807),
        'The Danforth': (43.6778, -79.3489),
        
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
        'Yorkdale': (43.7252, -79.4522)
    }

def quick_geocode_dataset(df):
    """Quickly add coordinates using district mapping."""
    print("🚀 QUICK GEOCODING WITH DISTRICT COORDINATES")
    print("=" * 47)
    
    # Get district coordinates
    district_coords = get_district_coordinates()
    
    # Add coordinate columns
    df['Latitude'] = np.nan
    df['Longitude'] = np.nan
    
    # Count records by address type
    toronto_only = (df['Address'] == 'Toronto').sum()
    detailed_addresses = (df['Address'] != 'Toronto').sum()
    
    print(f"📊 Address breakdown:")
    print(f"   'Toronto' only: {toronto_only:,} records")
    print(f"   Detailed addresses: {detailed_addresses:,} records")
    
    # Geocode using districts
    geocoded_count = 0
    
    for idx, row in df.iterrows():
        district = row.get('district')
        
        if pd.notna(district) and district in district_coords:
            lat, lon = district_coords[district]
            df.loc[idx, 'Latitude'] = lat
            df.loc[idx, 'Longitude'] = lon
            geocoded_count += 1
        elif pd.notna(district):
            # Try partial match
            district_lower = district.lower()
            for key, coords in district_coords.items():
                if district_lower in key.lower() or key.lower() in district_lower:
                    df.loc[idx, 'Latitude'] = coords[0]
                    df.loc[idx, 'Longitude'] = coords[1]
                    geocoded_count += 1
                    break
            else:
                # Default to downtown Toronto
                df.loc[idx, 'Latitude'] = 43.6532
                df.loc[idx, 'Longitude'] = -79.3832
                geocoded_count += 1
        else:
            # No district - default to downtown Toronto
            df.loc[idx, 'Latitude'] = 43.6532
            df.loc[idx, 'Longitude'] = -79.3832
            geocoded_count += 1
    
    coverage = geocoded_count / len(df) * 100
    
    print(f"\n✅ Geocoding complete:")
    print(f"   Total coordinates added: {geocoded_count:,}")
    print(f"   Coverage: {coverage:.1f}%")
    
    return df

def main():
    """Main function."""
    print("🧪 QUICK HOTEL GEOCODING TEST")
    print("=" * 32)
    
    # Load dataset
    df = pd.read_csv('Data/toronto_unified_hotel_analysis_districts_filled.csv')
    print(f"📂 Loaded dataset: {len(df):,} rows")
    
    # Quick geocoding
    df_with_coords = quick_geocode_dataset(df.copy())
    
    # Show coordinate coverage by district
    print(f"\n📍 Coordinate coverage by district:")
    district_summary = df_with_coords.groupby('district').agg({
        'Latitude': ['count', lambda x: x.notna().sum()],
        'Longitude': ['count', lambda x: x.notna().sum()]
    }).round(2)
    
    coord_coverage = df_with_coords[['Latitude', 'Longitude']].notna().all(axis=1)
    
    print(f"   Districts with coordinates: {df_with_coords['district'].nunique()}")
    print(f"   Records with coordinates: {coord_coverage.sum():,} / {len(df_with_coords):,}")
    
    # Show sample coordinates
    print(f"\n📋 Sample coordinates by district:")
    sample_districts = ['Bloor-Yorkville', 'Entertainment District', 'Airport', 'Downtown Toronto']
    
    for district in sample_districts:
        district_data = df_with_coords[df_with_coords['district'] == district]
        if len(district_data) > 0:
            sample = district_data.iloc[0]
            print(f"   {district}: ({sample['Latitude']:.4f}, {sample['Longitude']:.4f})")
    
    # Save with coordinates
    output_file = 'Data/toronto_hotels_with_coordinates.csv'
    df_with_coords.to_csv(output_file, index=False)
    print(f"\n✅ Dataset with coordinates saved to: {output_file}")
    
    return df_with_coords

if __name__ == "__main__":
    main()