"""
Fix Missing Distance Calculations for Hotel Dataset
=================================================

This script calculates missing "Distance from attraction" values for hotels
that have coordinates but missing distance data. Based on analysis, the main
attraction appears to be Rogers Centre (formerly SkyDome) in Toronto.

Key Toronto Attractions for Distance Calculation:
- Rogers Centre: 43.6414, -79.3894 (main stadium/entertainment venue)
- CN Tower: 43.6426, -79.3871 (iconic landmark)
- Union Station: 43.6452, -79.3806 (transportation hub)

Author: Spatial Analysis Fix
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import warnings
warnings.filterwarnings('ignore')

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in miles.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # Radius of earth in miles
    r = 3959
    
    return r * c

class DistanceFixer:
    def __init__(self, data_file='Data/toronto_hotels_transformed.csv'):
        """Initialize distance fixer."""
        self.data_file = data_file
        self.df = None
        
        # Key Toronto attractions (lat, lon)
        self.attractions = {
            'Rogers Centre': (43.6414, -79.3894),  # Main stadium
            'CN Tower': (43.6426, -79.3871),       # Iconic landmark  
            'Union Station': (43.6452, -79.3806),  # Transportation hub
            'Royal Ontario Museum': (43.6677, -79.3948),  # Cultural attraction
            'Casa Loma': (43.6780, -79.4103)       # Historic site
        }
        
        print("🔧 FIXING MISSING HOTEL DISTANCES")
        print("=" * 40)
        
    def load_data(self):
        """Load the dataset and analyze missing distances."""
        print("📂 Loading dataset...")
        
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"   ✅ Loaded {len(self.df):,} rows × {len(self.df.columns)} columns")
            
            # Analyze distance data
            total_hotels = len(self.df)
            missing_distance = self.df['Distance from attraction'].isnull().sum()
            available_distance = total_hotels - missing_distance
            
            print(f"\n📊 Distance Data Analysis:")
            print(f"   • Total hotels: {total_hotels:,}")
            print(f"   • With distances: {available_distance:,} ({available_distance/total_hotels*100:.1f}%)")
            print(f"   • Missing distances: {missing_distance:,} ({missing_distance/total_hotels*100:.1f}%)")
            
            return True
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
    
    def analyze_existing_distances(self):
        """Analyze existing distance calculations to understand the main attraction."""
        print(f"\n🔍 ANALYZING EXISTING DISTANCES")
        print("-" * 35)
        
        # Get hotels with existing distances
        with_distances = self.df[self.df['Distance from attraction'].notnull()].copy()
        
        if len(with_distances) == 0:
            print(f"   ⚠️ No existing distances found!")
            return None
        
        # Calculate distances to various attractions for comparison
        print(f"   📏 Testing distance calculations to major attractions...")
        
        # Sample a few hotels to test
        sample_hotels = with_distances.head(100)
        
        best_attraction = None
        best_correlation = -1
        
        for attraction_name, (attr_lat, attr_lon) in self.attractions.items():
            # Calculate distances to this attraction
            calculated_distances = []
            actual_distances = []
            
            for idx, hotel in sample_hotels.iterrows():
                if pd.notnull(hotel['latitude']) and pd.notnull(hotel['longitude']):
                    calc_dist = haversine_distance(
                        hotel['latitude'], hotel['longitude'],
                        attr_lat, attr_lon
                    )
                    calculated_distances.append(calc_dist)
                    actual_distances.append(hotel['Distance from attraction'])
            
            if len(calculated_distances) > 10:
                # Calculate correlation
                correlation = np.corrcoef(calculated_distances, actual_distances)[0, 1]
                
                print(f"     • {attraction_name:<20}: correlation = {correlation:.3f}")
                
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_attraction = (attraction_name, self.attractions[attraction_name])
        
        if best_attraction:
            print(f"\n   🎯 Best match: {best_attraction[0]} (correlation: {best_correlation:.3f})")
            print(f"   📍 Coordinates: {best_attraction[1]}")
            
            return best_attraction
        else:
            print(f"   ⚠️ No good correlation found, using Rogers Centre as default")
            return ('Rogers Centre', self.attractions['Rogers Centre'])
    
    def calculate_missing_distances(self, main_attraction):
        """Calculate distances for hotels missing this data."""
        print(f"\n🔨 CALCULATING MISSING DISTANCES")
        print("-" * 35)
        
        attraction_name, (attr_lat, attr_lon) = main_attraction
        print(f"   🎯 Using {attraction_name} as reference point")
        print(f"   📍 Coordinates: ({attr_lat}, {attr_lon})")
        
        # Get hotels with missing distances
        missing_mask = self.df['Distance from attraction'].isnull()
        missing_hotels = self.df[missing_mask].copy()
        
        print(f"   📊 Hotels to process: {len(missing_hotels):,}")
        
        # Calculate distances
        calculated_count = 0
        skipped_count = 0
        
        for idx in missing_hotels.index:
            hotel_lat = self.df.loc[idx, 'latitude']
            hotel_lon = self.df.loc[idx, 'longitude']
            
            if pd.notnull(hotel_lat) and pd.notnull(hotel_lon):
                # Calculate distance
                distance = haversine_distance(hotel_lat, hotel_lon, attr_lat, attr_lon)
                
                # Update the dataframe
                self.df.loc[idx, 'Distance from attraction'] = distance
                calculated_count += 1
            else:
                skipped_count += 1
        
        print(f"   ✅ Calculated distances: {calculated_count:,}")
        if skipped_count > 0:
            print(f"   ⚠️ Skipped (missing coordinates): {skipped_count:,}")
        
        # Verify results
        remaining_missing = self.df['Distance from attraction'].isnull().sum()
        print(f"   📈 Remaining missing distances: {remaining_missing:,}")
        
        return calculated_count
    
    def validate_calculations(self):
        """Validate the distance calculations."""
        print(f"\n✅ VALIDATION")
        print("-" * 15)
        
        # Distance statistics
        distances = self.df['Distance from attraction'].dropna()
        
        print(f"   📊 Distance Statistics:")
        print(f"     • Total hotels with distances: {len(distances):,}")
        print(f"     • Min distance: {distances.min():.2f} miles")
        print(f"     • Max distance: {distances.max():.2f} miles")
        print(f"     • Mean distance: {distances.mean():.2f} miles")
        print(f"     • Median distance: {distances.median():.2f} miles")
        
        # Check for reasonable values (Toronto metropolitan area)
        reasonable_distances = distances[(distances >= 0) & (distances <= 50)]
        unreasonable_count = len(distances) - len(reasonable_distances)
        
        print(f"   🎯 Reasonableness Check:")
        print(f"     • Reasonable distances (0-50 miles): {len(reasonable_distances):,}")
        if unreasonable_count > 0:
            print(f"     • Potentially unreasonable: {unreasonable_count:,}")
        else:
            print(f"     • All distances look reasonable ✅")
    
    def save_fixed_data(self):
        """Save the dataset with fixed distances."""
        print(f"\n💾 SAVING FIXED DATA")
        print("-" * 20)
        
        # Create output filename
        output_file = 'Data/toronto_hotels_distances_fixed.csv'
        
        try:
            self.df.to_csv(output_file, index=False)
            print(f"   ✅ Saved fixed dataset to: {output_file}")
            
            # Also update the original transformed file
            self.df.to_csv(self.data_file, index=False)
            print(f"   ✅ Updated original file: {self.data_file}")
            
            return output_file
        except Exception as e:
            print(f"   ❌ Error saving data: {e}")
            return None
    
    def run_distance_fix(self):
        """Run the complete distance fixing process."""
        print(f"🚀 Starting Distance Fix Process...")
        print()
        
        # Load data
        if not self.load_data():
            return False
        
        # Analyze existing distances to find the main attraction
        main_attraction = self.analyze_existing_distances()
        
        if not main_attraction:
            print(f"❌ Could not determine main attraction!")
            return False
        
        # Calculate missing distances
        calculated_count = self.calculate_missing_distances(main_attraction)
        
        if calculated_count == 0:
            print(f"❌ No distances were calculated!")
            return False
        
        # Validate results
        self.validate_calculations()
        
        # Save fixed data
        output_file = self.save_fixed_data()
        
        if output_file:
            print(f"\n🎉 DISTANCE FIX COMPLETED!")
            print(f"   📊 Fixed {calculated_count:,} missing distances")
            print(f"   💾 Saved to: {output_file}")
            return True
        else:
            return False


def main():
    """Run the distance fixing process."""
    
    # Initialize distance fixer
    fixer = DistanceFixer('Data/toronto_hotels_transformed.csv')
    
    # Run the fix
    success = fixer.run_distance_fix()
    
    if success:
        print(f"\n✅ Distance fixing completed successfully!")
        print(f"📈 Dataset is now ready for feature engineering with complete distance data")
    else:
        print(f"\n❌ Distance fixing failed!")
    
    return success


if __name__ == "__main__":
    main()