"""
Demonstrate Spatial Analysis with Real Hotel Coordinates

This script shows how the spatial analysis works with real hotel coordinates
and events, demonstrating the improvements over fixed placeholder distances.
"""

import pandas as pd
from spatial_analysis import (
    calculate_real_distances,
    find_events_during_stay_with_distances,
    create_spatial_event_features,
    analyze_spatial_patterns
)

def load_sample_data():
    """Load the sample data with coordinates."""
    print("📂 Loading sample data...")
    
    try:
        hotels_df = pd.read_csv('Data/sample_hotels_with_coordinates.csv')
        events_df = pd.read_csv('Data/sample_events.csv')
        
        print(f"   ✅ Loaded {len(hotels_df)} hotels and {len(events_df)} events")
        
        # Show coordinate availability
        coords_available = hotels_df['hotel_latitude'].notna().sum()
        print(f"   📍 Hotels with coordinates: {coords_available}/{len(hotels_df)}")
        
        return hotels_df, events_df
    
    except FileNotFoundError as e:
        print(f"   ❌ Sample data not found: {e}")
        return None, None

def demonstrate_distance_calculation(hotels_df, events_df):
    """Show the difference between placeholder and real distances."""
    print("\n🔍 COMPARING DISTANCE METHODS")
    print("=" * 50)
    
    # Filter to hotels with coordinates
    hotels_with_coords = hotels_df.dropna(subset=['hotel_latitude', 'hotel_longitude'])
    
    if len(hotels_with_coords) == 0:
        print("❌ No hotels with coordinates available")
        return None
    
    print(f"📊 Analyzing {len(hotels_with_coords)} hotels with coordinates")
    
    # Show hotel locations
    print(f"\n🏨 HOTEL LOCATIONS:")
    for _, hotel in hotels_with_coords.iterrows():
        district = hotel.get('district', 'Unknown')
        print(f"   {hotel['Hotel name'][:40]} ({district}): ({hotel['hotel_latitude']:.4f}, {hotel['hotel_longitude']:.4f})")
    
    # Calculate real distances
    distances_df = calculate_real_distances(hotels_with_coords, events_df)
    
    if distances_df.empty:
        print("❌ Failed to calculate distances")
        return None
    
    # Show distance insights
    print(f"\n📍 DISTANCE INSIGHTS:")
    print(f"   Distance range: {distances_df['distance_km'].min():.2f} - {distances_df['distance_km'].max():.2f} km")
    print(f"   Average distance: {distances_df['distance_km'].mean():.2f} km")
    print(f"   Median distance: {distances_df['distance_km'].median():.2f} km")
    
    # Show hotel-specific distance patterns
    print(f"\n🏨 HOTEL-SPECIFIC DISTANCE PATTERNS:")
    for hotel_idx in hotels_with_coords.index[:3]:  # Show first 3 hotels
        hotel_name = hotels_with_coords.loc[hotel_idx, 'Hotel name']
        hotel_distances = distances_df[distances_df['hotel_index'] == hotel_idx]
        
        if len(hotel_distances) > 0:
            min_dist = hotel_distances['distance_km'].min()
            max_dist = hotel_distances['distance_km'].max()
            avg_dist = hotel_distances['distance_km'].mean()
            
            print(f"   {hotel_name[:40]}:")
            print(f"      Min distance to event: {min_dist:.2f} km")
            print(f"      Max distance to event: {max_dist:.2f} km") 
            print(f"      Avg distance to events: {avg_dist:.2f} km")
    
    return distances_df

def demonstrate_spatial_features(hotels_df, events_df, distances_df):
    """Demonstrate the creation of spatial features."""
    print("\n🎯 SPATIAL FEATURE DEMONSTRATION")
    print("=" * 50)
    
    # Ensure date columns are datetime
    hotels_df['Check-in Date'] = pd.to_datetime(hotels_df['Check-in Date'])
    hotels_df['Check-out Date'] = pd.to_datetime(hotels_df['Check-out Date'])
    events_df['Date'] = pd.to_datetime(events_df['Date'])
    
    # Find events during stays with distances
    stay_events = find_events_during_stay_with_distances(hotels_df, events_df, distances_df)
    
    if stay_events.empty:
        print("❌ No events found during hotel stays")
        return
    
    print(f"✅ Found {len(stay_events)} events during hotel stays")
    
    # Show examples of events during stays with distances
    print(f"\n🎭 EVENTS DURING HOTEL STAYS (with distances):")
    for hotel_idx in stay_events['hotel_index'].unique()[:3]:  # Show first 3 hotels
        hotel_events = stay_events[stay_events['hotel_index'] == hotel_idx]
        hotel_name = hotel_events.iloc[0]['hotel_name']
        checkin = hotel_events.iloc[0]['hotel_checkin'].strftime('%m-%d')
        checkout = hotel_events.iloc[0]['hotel_checkout'].strftime('%m-%d')
        
        print(f"\n   🏨 {hotel_name[:40]} ({checkin} to {checkout}):")
        for _, event in hotel_events.head(5).iterrows():  # Show up to 5 events
            event_date = event['Date'].strftime('%m-%d')
            distance = event.get('distance_km', 'N/A')
            score = event.get('event_score', 0)
            print(f"      {event_date}: {event['Event Name'][:35]} - {distance} km (score: {score})")
    
    # Create spatial features
    spatial_features = create_spatial_event_features(stay_events)
    
    if spatial_features.empty:
        print("❌ Failed to create spatial features")
        return
    
    print(f"\n📊 SPATIAL FEATURES CREATED:")
    print(f"   Hotels with spatial features: {len(spatial_features)}")
    print(f"   Spatial feature columns: {len(spatial_features.columns)-1}")
    
    # Show key spatial features for each hotel
    print(f"\n🔍 KEY SPATIAL FEATURES BY HOTEL:")
    for _, features in spatial_features.iterrows():
        hotel_idx = features['hotel_index']
        hotel_name = hotels_df.loc[hotel_idx, 'Hotel name']
        
        print(f"\n   🏨 {hotel_name[:40]}:")
        print(f"      Events during stay: {features['events_count']}")
        print(f"      Avg distance to events: {features['avg_distance_to_events']:.2f} km")
        print(f"      Closest event: {features['min_distance_to_event']:.2f} km")
        print(f"      Distance-weighted score: {features['distance_weighted_event_score']:.2f}")
        print(f"      Has events within 1km: {'Yes' if features['has_nearby_event_1km'] else 'No'}")
        print(f"      Has events within 5km: {'Yes' if features['has_nearby_event_5km'] else 'No'}")
        print(f"      Event accessibility score: {features['event_accessibility_score']:.2f}")
    
    # Analyze patterns
    analyze_spatial_patterns(hotels_df, spatial_features)
    
    return spatial_features

def main():
    """Main demonstration function."""
    print("🌟 SPATIAL ANALYSIS DEMONSTRATION")
    print("=" * 60)
    print("This demo shows how real hotel coordinates enable meaningful")
    print("spatial analysis instead of placeholder distance values.")
    print("=" * 60)
    
    # Load sample data
    hotels_df, events_df = load_sample_data()
    
    if hotels_df is None or events_df is None:
        print("❌ Cannot proceed without sample data")
        return
    
    # Demonstrate distance calculation
    distances_df = demonstrate_distance_calculation(hotels_df, events_df)
    
    if distances_df is None:
        print("❌ Cannot proceed without distance data")
        return
    
    # Demonstrate spatial features
    spatial_features = demonstrate_spatial_features(hotels_df, events_df, distances_df)
    
    if spatial_features is not None:
        print(f"\n🎉 SPATIAL ANALYSIS COMPLETE!")
        print(f"✅ Successfully created {len(spatial_features.columns)-1} spatial features")
        print(f"📍 Based on real distances between {len(hotels_df)} hotels and {len(events_df)} events")
        print(f"💡 These features will help the model learn location-specific pricing patterns!")

if __name__ == "__main__":
    main()