"""
Test Spatial Analysis

This script creates a small sample with mock coordinates to test the spatial analysis functions.
"""

import pandas as pd
import numpy as np
from spatial_analysis import (
    calculate_real_distances,
    find_events_during_stay_with_distances,
    create_spatial_event_features
)

def create_test_data():
    """Create test data with mock coordinates."""
    print("🧪 Creating test data with mock coordinates...")
    
    # Create sample hotel data with coordinates (downtown Toronto area)
    hotels_test = pd.DataFrame({
        'Hotel name': [
            'Test Hotel Downtown',
            'Test Hotel Midtown', 
            'Test Hotel Airport'
        ],
        'Check-in Date': pd.to_datetime(['2025-10-13', '2025-10-14', '2025-10-15']),
        'Check-out Date': pd.to_datetime(['2025-10-15', '2025-10-16', '2025-10-17']),
        'price_cad': [200, 150, 100],
        'length_of_stay': [2, 2, 2],
        'hotel_latitude': [43.6532, 43.6570, 43.6436],  # Downtown, Midtown, Airport
        'hotel_longitude': [-79.3832, -79.3900, -79.5656]
    })
    
    # Create sample events data with coordinates
    events_test = pd.DataFrame({
        'Event Name': [
            'Concert at Rogers Centre',
            'Sports Game at Scotiabank Arena',
            'Festival at Harbourfront',
            'Conference at Convention Centre',
            'Show at Roy Thomson Hall'
        ],
        'Date': pd.to_datetime([
            '2025-10-13',  # During hotel 1 stay
            '2025-10-14',  # During hotel 1 & 2 stay
            '2025-10-15',  # During hotel 2 & 3 stay
            '2025-10-16',  # During hotel 2 & 3 stay
            '2025-10-17'   # Not during any stay (checkout day)
        ]),
        'Latitude': [43.6414, 43.6434, 43.6391, 43.6426, 43.6465],
        'Longitude': [-79.3894, -79.3791, -79.3799, -79.3775, -79.3871],
        'event_score': [12.0, 11.2, 8.5, 7.0, 9.5],
        'Segment': ['Music', 'Sports', 'Arts', 'Business', 'Music']
    })
    
    print(f"   ✅ Created {len(hotels_test)} test hotels and {len(events_test)} test events")
    return hotels_test, events_test

def test_spatial_functions():
    """Test the spatial analysis functions."""
    print("\n🧪 TESTING SPATIAL ANALYSIS FUNCTIONS")
    print("=" * 50)
    
    # Create test data
    hotels_test, events_test = create_test_data()
    
    # Test 1: Calculate real distances
    print("\n1️⃣ Testing distance calculation...")
    distances_df = calculate_real_distances(hotels_test, events_test)
    
    if not distances_df.empty:
        print(f"   ✅ Distance calculation successful: {len(distances_df)} pairs")
        print(f"   📊 Distance range: {distances_df['distance_km'].min():.2f} - {distances_df['distance_km'].max():.2f} km")
        
        # Show sample distances
        print("\n   📍 Sample distances:")
        sample_distances = distances_df.head(5)
        for _, row in sample_distances.iterrows():
            print(f"      {row['hotel_name'][:20]} -> {row['event_name'][:25]}: {row['distance_km']:.2f} km")
    else:
        print("   ❌ Distance calculation failed")
        return
    
    # Test 2: Find events during stays with distances
    print("\n2️⃣ Testing stay-based event matching with distances...")
    stay_events = find_events_during_stay_with_distances(hotels_test, events_test, distances_df)
    
    if not stay_events.empty:
        print(f"   ✅ Stay event matching successful: {len(stay_events)} events during stays")
        
        # Show events by hotel
        for hotel_idx in stay_events['hotel_index'].unique():
            hotel_events = stay_events[stay_events['hotel_index'] == hotel_idx]
            hotel_name = hotel_events.iloc[0]['hotel_name']
            checkin = hotel_events.iloc[0]['hotel_checkin'].strftime('%m-%d')
            checkout = hotel_events.iloc[0]['hotel_checkout'].strftime('%m-%d')
            
            print(f"\n   🏨 {hotel_name} ({checkin} to {checkout}):")
            for _, event in hotel_events.iterrows():
                event_date = event['Date'].strftime('%m-%d')
                distance = event.get('distance_km', 'N/A')
                score = event.get('event_score', 0)
                print(f"      {event_date}: {event['Event Name'][:30]} ({distance} km, score: {score})")
    else:
        print("   ❌ Stay event matching failed")
        return
    
    # Test 3: Create spatial features
    print("\n3️⃣ Testing spatial feature creation...")
    spatial_features = create_spatial_event_features(stay_events)
    
    if not spatial_features.empty:
        print(f"   ✅ Spatial feature creation successful: {len(spatial_features)} hotel records")
        print(f"   📊 Created {len(spatial_features.columns)-1} spatial features")
        
        # Show sample features
        print("\n   📋 Sample spatial features:")
        for _, hotel in spatial_features.iterrows():
            print(f"      Hotel {hotel['hotel_index']}:")
            print(f"         Events during stay: {hotel['events_count']}")
            print(f"         Avg distance to events: {hotel['avg_distance_to_events']:.2f} km")
            print(f"         Distance-weighted score: {hotel['distance_weighted_event_score']:.2f}")
            print(f"         Has nearby events (1km): {bool(hotel['has_nearby_event_1km'])}")
            print(f"         Event accessibility score: {hotel['event_accessibility_score']:.2f}")
            print()
    else:
        print("   ❌ Spatial feature creation failed")
        return
    
    print("🎉 All spatial analysis tests passed!")
    return True

if __name__ == "__main__":
    test_spatial_functions()