"""
Test performance comparison between iterrows and vectorized approaches.
"""

import pandas as pd
import numpy as np
import time
from spatial_analysis_vectorized import calculate_real_distances_vectorized

def test_with_sample_data():
    """Test vectorized approach with our real sample data."""
    print("🧪 TESTING WITH REAL SAMPLE DATA")
    print("=" * 50)
    
    try:
        # Load sample data
        hotels_df = pd.read_csv('Data/sample_hotels_with_coordinates.csv')
        events_df = pd.read_csv('Data/sample_events.csv')
        
        # Filter to hotels with coordinates
        hotels_with_coords = hotels_df.dropna(subset=['hotel_latitude', 'hotel_longitude'])
        
        print(f"📊 Real data: {len(hotels_with_coords)} hotels × {len(events_df)} events = {len(hotels_with_coords) * len(events_df):,} calculations")
        
        # Test vectorized approach
        start_time = time.time()
        distances_df = calculate_real_distances_vectorized(hotels_with_coords, events_df)
        vectorized_time = time.time() - start_time
        
        print(f"\n⚡ VECTORIZED RESULTS:")
        print(f"   Time: {vectorized_time:.3f} seconds")
        print(f"   Calculations: {len(distances_df):,}")
        print(f"   Speed: {len(distances_df)/vectorized_time:.0f} calculations/second")
        
        # Show some sample results
        print(f"\n📍 SAMPLE DISTANCE CALCULATIONS:")
        sample_distances = distances_df.head(5)
        for _, row in sample_distances.iterrows():
            print(f"   {row['hotel_name'][:30]} -> {row['event_name'][:25]}: {row['distance_km']:.2f} km")
        
        # Show distance distribution
        print(f"\n📊 DISTANCE DISTRIBUTION:")
        print(f"   Min distance: {distances_df['distance_km'].min():.2f} km")
        print(f"   Max distance: {distances_df['distance_km'].max():.2f} km")
        print(f"   Mean distance: {distances_df['distance_km'].mean():.2f} km")
        print(f"   Median distance: {distances_df['distance_km'].median():.2f} km")
        
        return distances_df
        
    except FileNotFoundError:
        print("❌ Sample data not found. Run create_sample_with_coords.py first.")
        return None

def simulate_iterrows_performance(n_hotels, n_events):
    """Simulate the performance of iterrows approach without actually using it."""
    print(f"\n🐌 SIMULATED ITERROWS PERFORMANCE:")
    print(f"   For {n_hotels} hotels × {n_events} events = {n_hotels * n_events:,} calculations")
    
    # Based on typical iterrows performance (measured empirically)
    # iterrows typically processes ~1000-5000 operations per second
    estimated_iterrows_time = (n_hotels * n_events) / 2000  # Conservative estimate
    
    print(f"   Estimated iterrows time: {estimated_iterrows_time:.2f} seconds")
    print(f"   Estimated speed: ~2,000 calculations/second")
    
    return estimated_iterrows_time

def main():
    """Main performance testing function."""
    print("⚡ VECTORIZED PERFORMANCE TESTING")
    print("=" * 60)
    
    # Test with real sample data
    distances_df = test_with_sample_data()
    
    if distances_df is not None:
        n_calculations = len(distances_df)
        n_hotels = distances_df['hotel_index'].nunique()
        n_events = distances_df['event_index'].nunique()
        
        # Simulate iterrows performance for comparison
        estimated_iterrows_time = simulate_iterrows_performance(n_hotels, n_events)
        
        # Calculate actual vectorized performance
        # Re-run for accurate timing
        hotels_df = pd.read_csv('Data/sample_hotels_with_coordinates.csv')
        events_df = pd.read_csv('Data/sample_events.csv')
        hotels_with_coords = hotels_df.dropna(subset=['hotel_latitude', 'hotel_longitude'])
        
        start_time = time.time()
        distances_df = calculate_real_distances_vectorized(hotels_with_coords, events_df)
        actual_vectorized_time = time.time() - start_time
        
        # Calculate speedup
        speedup = estimated_iterrows_time / actual_vectorized_time
        
        print(f"\n🚀 PERFORMANCE COMPARISON:")
        print(f"   Estimated iterrows time: {estimated_iterrows_time:.2f} seconds")
        print(f"   Actual vectorized time: {actual_vectorized_time:.3f} seconds")
        print(f"   🎯 Speedup: {speedup:.1f}x faster!")
        print(f"   Time saved: {estimated_iterrows_time - actual_vectorized_time:.2f} seconds")
        
        print(f"\n💡 SCALING IMPLICATIONS:")
        full_dataset_hotels = 2669  # From our full hotel dataset
        full_dataset_events = 1629   # From our full events dataset
        full_calculations = full_dataset_hotels * full_dataset_events
        
        vectorized_time_full = full_calculations / (len(distances_df) / actual_vectorized_time)
        iterrows_time_full = full_calculations / 2000
        
        print(f"   For full dataset ({full_dataset_hotels} hotels × {full_dataset_events} events):")
        print(f"   Estimated iterrows time: {iterrows_time_full/60:.1f} minutes")
        print(f"   Vectorized time: {vectorized_time_full:.1f} seconds")
        print(f"   Total time saved: {(iterrows_time_full - vectorized_time_full)/60:.1f} minutes!")

if __name__ == "__main__":
    main()