"""
Data Joining Module for Toronto Hotel Analysis

This module combines hotel, events, and weather data to create a unified dataset
for comprehensive analysis of hotel pricing and demand patterns in Toronto.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

# Import our spatial analysis module
from spatial_analysis import (
    calculate_real_distances, 
    find_events_during_stay_with_distances,
    create_spatial_event_features,
    analyze_spatial_patterns
)

def load_cleaned_datasets():
    """Load all cleaned datasets from the Data directory."""
    data_dir = 'Data'
    datasets = {}
    
    # Load hotel data - prefer geocoded version if available
    geocoded_hotel_file = os.path.join(data_dir, 'toronto_hotels_with_coordinates.csv')
    regular_hotel_file = os.path.join(data_dir, 'toronto_hotels_cleaned_transformed.csv')
    
    if os.path.exists(geocoded_hotel_file):
        datasets['hotels'] = pd.read_csv(geocoded_hotel_file)
        print(f"✅ Loaded geocoded hotel data: {len(datasets['hotels'])} records")
        # Check if coordinates are available
        if 'hotel_latitude' in datasets['hotels'].columns:
            coords_available = datasets['hotels']['hotel_latitude'].notna().sum()
            print(f"   📍 Hotels with coordinates: {coords_available}/{len(datasets['hotels'])}")
        else:
            print(f"   ⚠️ No coordinate columns found in geocoded file")
    elif os.path.exists(regular_hotel_file):
        datasets['hotels'] = pd.read_csv(regular_hotel_file)
        print(f"✅ Loaded hotel data: {len(datasets['hotels'])} records")
        # Check if coordinates are available in regular file
        if 'hotel_latitude' in datasets['hotels'].columns:
            coords_available = datasets['hotels']['hotel_latitude'].notna().sum()
            print(f"   📍 Hotels with coordinates: {coords_available}/{len(datasets['hotels'])}")
        else:
            print(f"   ⚠️ No coordinates available - spatial analysis will be limited")
    else:
        print(f"❌ Hotel data not found")
    
    # Load events data
    events_file = os.path.join(data_dir, 'toronto_events_cleaned.csv')
    if os.path.exists(events_file):
        datasets['events'] = pd.read_csv(events_file)
        print(f"✅ Loaded events data: {len(datasets['events'])} records")
    else:
        print(f"❌ Events data not found: {events_file}")
    
    # Load weather data
    weather_file = os.path.join(data_dir, 'toronto_weather_cleaned.csv')
    if os.path.exists(weather_file):
        datasets['weather'] = pd.read_csv(weather_file)
        print(f"✅ Loaded weather data: {len(datasets['weather'])} records")
    else:
        print(f"❌ Weather data not found: {weather_file}")
    
    return datasets

def prepare_data_for_joining(datasets):
    """Prepare datasets for joining by ensuring proper date formats and data types."""
    print("\n🔄 Preparing data for joining...")
    
    # Prepare hotel data
    if 'hotels' in datasets:
        hotels = datasets['hotels'].copy()
        
        # Convert date columns to datetime
        date_columns = ['Date', 'Check-in Date', 'Check-out Date']
        for col in date_columns:
            if col in hotels.columns:
                hotels[col] = pd.to_datetime(hotels[col], errors='coerce')
        
        # Ensure numeric columns are proper types
        numeric_columns = ['price_usd', 'price_cad', 'booking_lead_time', 'length_of_stay']
        for col in numeric_columns:
            if col in hotels.columns:
                hotels[col] = pd.to_numeric(hotels[col], errors='coerce')
        
        datasets['hotels'] = hotels
        print(f"   ✅ Hotel data prepared: {len(hotels)} records")
    
    # Prepare events data
    if 'events' in datasets:
        events = datasets['events'].copy()
        
        # Convert date column to datetime
        if 'Date' in events.columns:
            events['Date'] = pd.to_datetime(events['Date'], errors='coerce')
        
        # Ensure numeric columns are proper types
        numeric_columns = ['Latitude', 'Longitude', 'event_score']
        for col in numeric_columns:
            if col in events.columns:
                events[col] = pd.to_numeric(events[col], errors='coerce')
        
        datasets['events'] = events
        print(f"   ✅ Events data prepared: {len(events)} records")
    
    # Prepare weather data
    if 'weather' in datasets:
        weather = datasets['weather'].copy()
        
        # Convert date column to datetime
        if 'LOCAL_DATE' in weather.columns:
            weather['LOCAL_DATE'] = pd.to_datetime(weather['LOCAL_DATE'], errors='coerce')
        
        # Ensure numeric columns are proper types
        numeric_columns = [
            'MEAN_TEMPERATURE', 'MIN_TEMPERATURE', 'MAX_TEMPERATURE',
            'TOTAL_PRECIPITATION', 'TOTAL_SNOW', 'SNOW_ON_GROUND',
            'SPEED_MAX_GUST', 'MAX_REL_HUMIDITY'
        ]
        for col in numeric_columns:
            if col in weather.columns:
                weather[col] = pd.to_numeric(weather[col], errors='coerce')
        
        datasets['weather'] = weather
        print(f"   ✅ Weather data prepared: {len(weather)} records")
    
    return datasets

def has_hotel_coordinates(hotels_df):
    """Check if hotel data has coordinate information."""
    return ('hotel_latitude' in hotels_df.columns and 
            'hotel_longitude' in hotels_df.columns and
            hotels_df['hotel_latitude'].notna().sum() > 0)

def join_hotels_with_weather(hotels_df, weather_df):
    """Join hotel data with weather data based on check-in dates."""
    print("\n🌤️ Joining hotels with weather data...")
    
    # Join on check-in date
    hotels_weather = hotels_df.merge(
        weather_df,
        left_on='Check-in Date',
        right_on='LOCAL_DATE',
        how='left',
        suffixes=('', '_weather')
    )
    
    # Drop duplicate date column
    if 'LOCAL_DATE' in hotels_weather.columns:
        hotels_weather = hotels_weather.drop('LOCAL_DATE', axis=1)
    
    print(f"   ✅ Joined hotels with weather: {len(hotels_weather)} records")
    print(f"   📊 Weather data available for {hotels_weather['MEAN_TEMPERATURE'].notna().sum()} hotels")
    
    return hotels_weather

def find_events_during_stay(hotels_df, events_df):
    """Find events that occur during the hotel stay period (check-in to check-out)."""
    print(f"\n🎯 Finding events occurring during hotel stay periods...")
    
    # Ensure we have required date columns
    required_hotel_cols = ['Check-in Date', 'Check-out Date']
    required_event_cols = ['Date']
    
    missing_hotel_cols = [col for col in required_hotel_cols if col not in hotels_df.columns]
    missing_event_cols = [col for col in required_event_cols if col not in events_df.columns]
    
    if missing_hotel_cols or missing_event_cols:
        print(f"❌ Missing required columns:")
        if missing_hotel_cols:
            print(f"   Hotel data: {missing_hotel_cols}")
        if missing_event_cols:
            print(f"   Event data: {missing_event_cols}")
        return pd.DataFrame()
    
    # Filter out rows with missing dates
    hotels_clean = hotels_df.dropna(subset=['Check-in Date', 'Check-out Date']).copy()
    events_clean = events_df.dropna(subset=['Date']).copy()
    
    print(f"   📊 Processing {len(hotels_clean)} hotel bookings and {len(events_clean)} events")
    
    # Create a list to store hotel-event matches
    hotel_events = []
    
    # Use vectorized operations where possible
    for hotel_idx, hotel in hotels_clean.iterrows():
        checkin_date = hotel['Check-in Date']
        checkout_date = hotel['Check-out Date']
        
        # Find events that occur during the stay (inclusive of check-in, exclusive of check-out)
        stay_events = events_clean[
            (events_clean['Date'] >= checkin_date) & 
            (events_clean['Date'] < checkout_date)
        ].copy()
        
        # Add hotel context to each matching event
        if len(stay_events) > 0:
            stay_events = stay_events.copy()
            stay_events['hotel_index'] = hotel_idx
            stay_events['hotel_name'] = hotel['Hotel name']
            stay_events['hotel_checkin'] = checkin_date
            stay_events['hotel_checkout'] = checkout_date
            stay_events['hotel_district'] = hotel.get('district', 'Unknown')
            stay_events['hotel_price_cad'] = hotel.get('price_cad', None)
            stay_events['length_of_stay'] = hotel.get('length_of_stay', 0)
            
            # Calculate days from check-in to event
            stay_events['days_from_checkin'] = (stay_events['Date'] - checkin_date).dt.days
            
            hotel_events.append(stay_events)
    
    # Combine all hotel-event pairs
    if hotel_events:
        hotel_events_df = pd.concat(hotel_events, ignore_index=True)
        print(f"   ✅ Found {len(hotel_events_df)} events occurring during hotel stays")
        print(f"   🏨 {hotel_events_df['hotel_index'].nunique()} hotels have events during their stay")
    else:
        hotel_events_df = pd.DataFrame()
        print(f"   ⚠️ No events found during any hotel stays")
    
    return hotel_events_df

def aggregate_stay_events(hotel_events_df):
    """Aggregate event features for each hotel booking based on events during stay."""
    print("\n📊 Aggregating events occurring during hotel stays...")
    
    if hotel_events_df.empty:
        print("   ⚠️ No events during hotel stays to aggregate")
        return pd.DataFrame()
    
    # Group by hotel booking and create comprehensive event features
    agg_functions = {
        'event_score': ['sum', 'max', 'mean', 'count'],
        'days_from_checkin': ['min', 'max', 'mean'],
        'Segment': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',  # Most common event type
        'Event Name': 'count'  # Count of events (same as event_score count, but clearer)
    }
    
    # Optional: Add specific event type counts if needed
    try:
        event_impact = hotel_events_df.groupby('hotel_index').agg(agg_functions).round(2)
        
        # Flatten column names with descriptive names
        event_impact.columns = [
            'events_total_score',       # Sum of all event scores during stay
            'events_max_score',         # Highest impact event during stay  
            'events_avg_score',         # Average event impact during stay
            'events_count',             # Number of events during stay
            'events_earliest_day',      # Days from checkin to earliest event
            'events_latest_day',        # Days from checkin to latest event  
            'events_avg_day',           # Average days from checkin to events
            'events_primary_segment',   # Most common type of event during stay
            'events_name_count'         # Redundant with events_count, will drop
        ]
        
        # Drop redundant column
        event_impact = event_impact.drop('events_name_count', axis=1)
        
        # Add additional derived features
        event_impact['events_span_days'] = event_impact['events_latest_day'] - event_impact['events_earliest_day']
        event_impact['events_density'] = event_impact['events_count'] / (event_impact['events_span_days'] + 1)  # Events per day
        
        # Create binary flags for high-impact stays
        event_impact['has_major_event'] = (event_impact['events_max_score'] >= 10).astype(int)
        event_impact['has_multiple_events'] = (event_impact['events_count'] >= 2).astype(int)
        
        event_impact = event_impact.reset_index()
        print(f"   ✅ Aggregated stay events for {len(event_impact)} hotel bookings")
        print(f"   📊 Features created: {len(event_impact.columns)-1}")  # -1 for hotel_index
        
        return event_impact
        
    except Exception as e:
        print(f"   ❌ Error aggregating events: {e}")
        return pd.DataFrame()

def create_unified_dataset(datasets):
    """Create a unified dataset combining all data sources with spatial analysis."""
    print("\n🔗 Creating unified dataset with spatial analysis...")
    
    if 'hotels' not in datasets:
        print("❌ Cannot create unified dataset without hotel data")
        return None
    
    # Start with hotel data
    unified_df = datasets['hotels'].copy()
    print(f"   📊 Starting with {len(unified_df)} hotel records")
    
    # Add weather data
    if 'weather' in datasets:
        unified_df = join_hotels_with_weather(unified_df, datasets['weather'])
        print(f"   🌤️ Added weather data")
    
    # Add event data with spatial analysis
    if 'events' in datasets:
        hotels_have_coords = has_hotel_coordinates(unified_df)
        
        if hotels_have_coords:
            print("\n📍 SPATIAL ANALYSIS MODE: Using real hotel coordinates")
            
            # Calculate real distances between all hotels and events
            distances_df = calculate_real_distances(unified_df, datasets['events'])
            
            if not distances_df.empty:
                # Find events during stays with distance information
                hotel_stay_events = find_events_during_stay_with_distances(
                    unified_df, datasets['events'], distances_df
                )
                
                if not hotel_stay_events.empty:
                    # Create spatial event features
                    spatial_event_features = create_spatial_event_features(hotel_stay_events)
                    
                    if not spatial_event_features.empty:
                        # Join spatial features with unified dataset
                        unified_df = unified_df.merge(
                            spatial_event_features,
                            left_index=True,
                            right_on='hotel_index',
                            how='left'
                        )
                        
                        # Fill missing spatial features with appropriate defaults
                        spatial_columns = [
                            'events_total_score', 'events_max_score', 'events_avg_score', 'events_count',
                            'min_distance_to_event', 'max_distance_to_event', 'avg_distance_to_events', 'median_distance_to_events',
                            'events_earliest_day', 'events_latest_day', 'events_avg_day',
                            'events_span_days', 'events_density', 'distance_weighted_event_score',
                            'distance_to_closest_major_event', 'has_nearby_event_1km', 'has_nearby_event_5km',
                            'has_major_event', 'event_accessibility_score'
                        ]
                        
                        # Fill numeric columns with 0
                        numeric_cols = [col for col in spatial_columns if col != 'events_primary_segment']
                        for col in numeric_cols:
                            if col in unified_df.columns:
                                unified_df[col] = unified_df[col].fillna(0)
                        
                        # Fill categorical column with 'None'
                        if 'events_primary_segment' in unified_df.columns:
                            unified_df['events_primary_segment'] = unified_df['events_primary_segment'].fillna('None')
                        
                        print(f"   🎯 Added spatial event features with real distances")
                        
                        # Analyze spatial patterns
                        analyze_spatial_patterns(unified_df, spatial_event_features)
                    else:
                        print(f"   ❌ Failed to create spatial event features")
                else:
                    print(f"   ⚠️ No events found during hotel stays with distance data")
            else:
                print(f"   ❌ Failed to calculate distances - falling back to basic event features")
                # Fallback to basic event features without distances
                hotel_stay_events = find_events_during_stay(unified_df, datasets['events'])
                if not hotel_stay_events.empty:
                    stay_event_features = aggregate_stay_events(hotel_stay_events)
                    # ... (same basic joining logic as before)
        else:
            print("\n⚠️ BASIC MODE: No hotel coordinates available")
            print("   💡 Run geocode_hotels.py first for spatial analysis")
            
            # Fallback to basic event features without spatial analysis
            hotel_stay_events = find_events_during_stay(unified_df, datasets['events'])
            
            if not hotel_stay_events.empty:
                stay_event_features = aggregate_stay_events(hotel_stay_events)
                
                # Join with unified dataset
                unified_df = unified_df.merge(
                    stay_event_features,
                    left_index=True,
                    right_on='hotel_index',
                    how='left'
                )
                
                # Fill missing event features with appropriate defaults
                event_columns = [
                    'events_total_score', 'events_max_score', 'events_avg_score', 'events_count',
                    'events_earliest_day', 'events_latest_day', 'events_avg_day', 
                    'events_span_days', 'events_density', 'has_major_event', 'has_multiple_events'
                ]
                
                # Fill numeric columns with 0
                numeric_event_cols = [col for col in event_columns if col != 'events_primary_segment']
                for col in numeric_event_cols:
                    if col in unified_df.columns:
                        unified_df[col] = unified_df[col].fillna(0)
                
                # Fill categorical column with 'None'
                if 'events_primary_segment' in unified_df.columns:
                    unified_df['events_primary_segment'] = unified_df['events_primary_segment'].fillna('None')
                
                print(f"   🎯 Added basic event features (no spatial analysis)")
                print(f"   📊 {stay_event_features.shape[0]} hotels have events during their stay")
            else:
                print(f"   ⚠️ No events found during any hotel stays")
    
    print(f"\n✅ Unified dataset created: {len(unified_df)} records, {len(unified_df.columns)} columns")
    
    return unified_df

def save_unified_dataset(unified_df, filename='toronto_unified_hotel_analysis.csv'):
    """Save the unified dataset to CSV."""
    if unified_df is None:
        print("❌ No unified dataset to save")
        return None
    
    output_path = os.path.join('Data', filename)
    
    try:
        unified_df.to_csv(output_path, index=False)
        print(f"✅ Unified dataset saved to: {output_path}")
        print(f"   📊 Records: {len(unified_df)}")
        print(f"   📊 Columns: {len(unified_df.columns)}")
        
        # Show column summary
        print(f"\n📋 Dataset Columns:")
        for i, col in enumerate(unified_df.columns, 1):
            non_null = unified_df[col].notna().sum()
            print(f"   {i:2d}. {col}: {non_null}/{len(unified_df)} non-null")
        
        return output_path
    except Exception as e:
        print(f"❌ Error saving unified dataset: {e}")
        return None

def main():
    """Main function to join all datasets."""
    print("🔗 TORONTO HOTEL DATA JOINING PIPELINE")
    print("=" * 50)
    
    # Step 1: Load datasets
    print("\n1️⃣ Loading cleaned datasets...")
    datasets = load_cleaned_datasets()
    
    if not datasets:
        print("❌ No datasets loaded. Please run data_transformation.py first.")
        return
    
    # Step 2: Prepare data
    datasets = prepare_data_for_joining(datasets)
    
    # Step 3: Create unified dataset
    unified_df = create_unified_dataset(datasets)
    
    # Step 4: Save unified dataset
    if unified_df is not None:
        save_path = save_unified_dataset(unified_df)
        
        if save_path:
            print(f"\n🎉 Data joining completed successfully!")
            print(f"📁 Unified dataset available at: {save_path}")
            print(f"💡 Ready for comprehensive hotel demand and pricing analysis!")
        else:
            print(f"\n❌ Failed to save unified dataset")
    else:
        print(f"\n❌ Failed to create unified dataset")

if __name__ == "__main__":
    main()