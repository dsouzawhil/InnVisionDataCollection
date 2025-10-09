"""
Automated Hotel Data Pipeline
============================

This script automates the complete hotel data processing pipeline:
1. Combines individual hotel CSV files from daily scraping
2. Runs spatial analysis joining with events and weather data
3. Creates the unified dataset ready for ML transformation

Usage: python hotel_data_pipeline.py
"""

import pandas as pd
import glob
import os
import subprocess
import sys
from datetime import datetime

def combine_hotel_files():
    """Step 1: Combine individual hotel CSV files into one dataset."""
    print("🔗 STEP 1: COMBINING HOTEL LISTING FILES")
    print("=" * 50)
    
    # Find all hotel CSV files
    pattern = 'Data/hotel_listing/Toronto_hotels*.csv'
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print("❌ No hotel CSV files found in Data/hotel_listing/")
        return False
    
    print(f"📁 Found {len(csv_files)} hotel files to combine")
    
    # Show sample files
    for file in sorted(csv_files)[:5]:
        print(f"   • {os.path.basename(file)}")
    if len(csv_files) > 5:
        print(f"   • ... and {len(csv_files) - 5} more files")
    
    # Combine all files
    all_hotels = []
    total_rows = 0
    
    print(f"\n📊 Reading and combining files...")
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            all_hotels.append(df)
            total_rows += len(df)
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
            continue
    
    if not all_hotels:
        print("❌ No valid hotel files to combine")
        return False
    
    # Combine all dataframes
    print(f"🔄 Combining {len(all_hotels)} dataframes...")
    combined_df = pd.concat(all_hotels, ignore_index=True)
    
    print(f"\n📈 COMBINATION RESULTS:")
    print(f"   Total rows before dedup: {total_rows:,}")
    print(f"   Combined dataset: {len(combined_df):,} rows")
    
    # Remove duplicates based on key columns
    key_columns = ['Hotel name', 'Check-in Date', 'Check-out Date', 'Room Type','Date']
    # Only use columns that exist
    existing_key_columns = [col for col in key_columns if col in combined_df.columns]
    
    if existing_key_columns:
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=existing_key_columns, keep='first')
        after_dedup = len(combined_df)
        print(f"   After deduplication: {after_dedup:,} rows")
        print(f"   Removed duplicates: {before_dedup - after_dedup:,}")
    
    # Save combined file
    output_file = 'Data/toronto_hotels_cleaned_transformed.csv'
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved combined hotel data: {output_file}")
    print(f"📊 Final dataset: {len(combined_df):,} rows × {len(combined_df.columns)} columns")
    
    if 'Check-in Date' in combined_df.columns:
        print(f"📅 Date range: {combined_df['Check-in Date'].min()} to {combined_df['Check-in Date'].max()}")
    
    return True

def run_data_joining():
    """Step 2: Run data_joining.py to create unified dataset with spatial analysis."""
    print("\n🔗 STEP 2: RUNNING SPATIAL DATA JOINING")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        'Data/toronto_hotels_cleaned_transformed.csv',
        'Data/toronto_events_cleaned.csv',
        'Data/toronto_weather_cleaned.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n💡 Make sure you have run:")
        print("   • Hotel scraping (creates hotel data)")
        print("   • Event data collection (geteventsdata.py)")
        print("   • Weather data collection (get_weather.py)")
        return False
    
    print("✅ All required files found")
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"   • {os.path.basename(file)} ({size:.1f} KB)")
    
    # Run data_joining.py
    print(f"\n🚀 Running data_joining.py...")
    try:
        result = subprocess.run([sys.executable, 'data_joining.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Data joining completed successfully!")
            if result.stdout:
                # Show last few lines of output
                output_lines = result.stdout.strip().split('\n')
                print("\n📋 Final output:")
                for line in output_lines[-5:]:
                    print(f"   {line}")
            
            # Check if output file was created
            output_file = 'Data/toronto_unified_hotel_analysis.csv'
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                print(f"\n✅ Created: {output_file}")
                print(f"📊 Unified dataset: {len(df):,} rows × {len(df.columns)} columns")
                return True
            else:
                print(f"❌ Expected output file not found: {output_file}")
                return False
        else:
            print(f"❌ Data joining failed with return code: {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Data joining timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running data joining: {e}")
        return False

def check_pipeline_prerequisites():
    """Check if all prerequisites are available for the pipeline."""
    print("🔍 CHECKING PIPELINE PREREQUISITES")
    print("=" * 40)
    
    # Check for hotel listing files
    hotel_files = glob.glob('Data/hotel_listing/Toronto_hotels*.csv')
    print(f"📁 Hotel listing files: {len(hotel_files)} found")
    
    # Check for events data
    events_files = [
        'Data/toronto_events_cleaned.csv',
        'Data/toronto_events_with_scores.csv'
    ]
    events_found = [f for f in events_files if os.path.exists(f)]
    print(f"🎭 Events data files: {len(events_found)}/{len(events_files)} found")
    for f in events_found:
        print(f"   ✅ {os.path.basename(f)}")
    
    # Check for weather data
    weather_files = [
        'Data/toronto_weather_cleaned.csv',
        'Data/toronto_weather_2025.csv'
    ]
    weather_found = [f for f in weather_files if os.path.exists(f)]
    print(f"🌤️ Weather data files: {len(weather_found)}/{len(weather_files)} found")
    for f in weather_found:
        print(f"   ✅ {os.path.basename(f)}")
    
    # Check for required Python files
    required_scripts = ['data_joining.py', 'spatial_analysis.py']
    scripts_found = [f for f in required_scripts if os.path.exists(f)]
    print(f"🐍 Required scripts: {len(scripts_found)}/{len(required_scripts)} found")
    for f in scripts_found:
        print(f"   ✅ {f}")
    
    # Overall readiness
    all_ready = (len(hotel_files) > 0 and 
                len(events_found) > 0 and 
                len(weather_found) > 0 and 
                len(scripts_found) == len(required_scripts))
    
    print(f"\n🚦 Pipeline readiness: {'✅ Ready' if all_ready else '❌ Not ready'}")
    
    return all_ready

def main():
    """Run the complete hotel data pipeline."""
    print("🏨 AUTOMATED HOTEL DATA PIPELINE")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_pipeline_prerequisites():
        print("\n❌ Prerequisites not met. Please ensure all required files are available.")
        return False
    
    print("\n🚀 Starting pipeline execution...")
    
    # Step 1: Combine hotel files
    if not combine_hotel_files():
        print("\n❌ Pipeline failed at Step 1: Hotel file combination")
        return False
    
    # Step 2: Run data joining
    if not run_data_joining():
        print("\n❌ Pipeline failed at Step 2: Data joining")
        return False
    
    # Success!
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("📁 Output files created:")
    print("   ✅ Data/toronto_hotels_cleaned_transformed.csv (combined hotel data)")
    print("   ✅ Data/toronto_unified_hotel_analysis.csv (unified with spatial features)")
    print("\n🎯 Next steps:")
    print("   1. Run data_transformation.py for ML-ready dataset")
    print("   2. Run quick_eda.py for exploratory data analysis")
    print("   3. Build your price prediction model!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)