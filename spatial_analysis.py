"""
Spatial Analysis Module for Toronto Hotel Analysis

This module provides functions to calculate real distances between hotels and events,
and create meaningful spatial features for hotel pricing analysis.

PERFORMANCE NOTE: This module has been optimized to eliminate slow iterrows() operations
and uses vectorized operations for production ML pipelines.
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

# Import vectorized functions for better performance
from spatial_analysis_vectorized import (
    calculate_real_distances_vectorized,
    find_events_during_stay_vectorized, 
    create_spatial_features_vectorized
)

def calculate_real_distances(hotels_df, events_df):
    """
    Calculate real distances between hotels and events using vectorized operations.
    
    Args:
        hotels_df: DataFrame with hotel data including hotel_latitude, hotel_longitude
        events_df: DataFrame with events data including Latitude, Longitude
    
    Returns:
        DataFrame with hotel-event distance pairs
    """
    # Use optimized vectorized implementation
    return calculate_real_distances_vectorized(hotels_df, events_df)

def find_events_during_stay_with_distances(hotels_df, events_df, distances_df):
    """
    Find events during hotel stays using vectorized operations.
    
    Args:
        hotels_df: DataFrame with hotel booking data
        events_df: DataFrame with events data
        distances_df: DataFrame with pre-calculated distances
    
    Returns:
        DataFrame with events during stays including distance information
    """
    # Use optimized vectorized implementation
    return find_events_during_stay_vectorized(hotels_df, events_df, distances_df)

def create_spatial_event_features(hotel_events_df):
    """
    Create spatial features using vectorized operations.
    
    Args:
        hotel_events_df: DataFrame with events during stays including distances
    
    Returns:
        DataFrame with aggregated spatial features per hotel booking
    """
    # Use optimized vectorized implementation
    return create_spatial_features_vectorized(hotel_events_df)

def analyze_spatial_patterns(hotels_df, spatial_features_df):
    """Analyze spatial patterns in the data."""
    print("\n🔍 SPATIAL ANALYSIS SUMMARY:")
    print("=" * 50)
    
    if spatial_features_df.empty:
        print("❌ No spatial features to analyze")
        return
    
    # Distance analysis
    print(f"📍 DISTANCE PATTERNS:")
    print(f"   Average distance to events: {spatial_features_df['avg_distance_to_events'].mean():.2f} km")
    print(f"   Closest event distance: {spatial_features_df['min_distance_to_event'].min():.2f} km")
    print(f"   Furthest event distance: {spatial_features_df['max_distance_to_event'].max():.2f} km")
    
    # Proximity analysis
    nearby_1km = spatial_features_df['has_nearby_event_1km'].sum()
    nearby_5km = spatial_features_df['has_nearby_event_5km'].sum()
    print(f"   Hotels with events within 1km: {nearby_1km} ({nearby_1km/len(spatial_features_df)*100:.1f}%)")
    print(f"   Hotels with events within 5km: {nearby_5km} ({nearby_5km/len(spatial_features_df)*100:.1f}%)")
    
    # Event impact analysis
    major_events = spatial_features_df['has_major_event'].sum()
    print(f"   Hotels with major events nearby: {major_events} ({major_events/len(spatial_features_df)*100:.1f}%)")
    
    # Distance-weighted scores
    print(f"   Average distance-weighted event score: {spatial_features_df['distance_weighted_event_score'].mean():.2f}")
    print(f"   Max event accessibility score: {spatial_features_df['event_accessibility_score'].max():.2f}")

def main():
    """Main function to demonstrate spatial analysis."""
    print("📍 SPATIAL ANALYSIS DEMO")
    print("=" * 30)
    
    # This would be called from the main data joining pipeline
    print("⚠️ This module is designed to be imported and used in data_joining.py")
    print("💡 Run the updated data_joining.py to see spatial analysis in action")

if __name__ == "__main__":
    main()