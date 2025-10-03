"""
Test the holiday flag function on the clean hotel dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Import the holiday function from data_transformation
sys.path.append('.')
from data_transformation import add_holiday_flag

def test_holiday_function():
    """Test the holiday flag function."""
    print("🧪 TESTING HOLIDAY FLAG FUNCTION")
    print("=" * 40)
    
    # Load the clean dataset
    df = pd.read_csv('Data/toronto_unified_hotel_analysis_clean.csv')
    print(f"📊 Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Show date range
    df['Check-in Date'] = pd.to_datetime(df['Check-in Date'])
    df['Check-out Date'] = pd.to_datetime(df['Check-out Date'])
    
    print(f"📅 Check-in date range: {df['Check-in Date'].min()} to {df['Check-in Date'].max()}")
    print(f"📅 Check-out date range: {df['Check-out Date'].min()} to {df['Check-out Date'].max()}")
    
    # Test holiday function
    df_with_holidays = add_holiday_flag(df)
    
    if df_with_holidays is not None and 'is_holiday' in df_with_holidays.columns:
        # Analyze results
        print(f"\n📊 HOLIDAY ANALYSIS RESULTS")
        print("=" * 30)
        
        holiday_stats = df_with_holidays['is_holiday'].value_counts()
        print(f"Holiday distribution:")
        for value, count in holiday_stats.items():
            pct = (count / len(df_with_holidays)) * 100
            status = "includes holidays" if value else "no holidays"
            print(f"  {status}: {count:,} bookings ({pct:.1f}%)")
        
        # Show some examples of holiday bookings
        holiday_bookings = df_with_holidays[df_with_holidays['is_holiday'] == True]
        
        if len(holiday_bookings) > 0:
            print(f"\n🎉 Sample holiday bookings:")
            sample_cols = ['Check-in Date', 'Check-out Date', 'Hotel name', 'price_cad', 'is_holiday']
            sample_cols = [col for col in sample_cols if col in holiday_bookings.columns]
            
            sample = holiday_bookings[sample_cols].head(10)
            print(sample.to_string(index=False))
            
            # Show which specific holidays are occurring
            print(f"\n📋 Canadian holidays in October 2025:")
            
            import holidays
            canada_holidays = holidays.Canada(prov='ON')
            
            # Get October 2025 holidays
            oct_2025_holidays = {date: name for date, name in canada_holidays.items() 
                               if date.year == 2025 and date.month == 10}
            
            if oct_2025_holidays:
                for date, name in oct_2025_holidays.items():
                    print(f"  {date}: {name}")
            else:
                print("  No holidays in October 2025")
                
                # Show some nearby holidays
                print(f"\n📋 Nearby holidays in 2025:")
                holidays_2025 = {date: name for date, name in canada_holidays.items() 
                               if date.year == 2025}
                
                for i, (date, name) in enumerate(list(holidays_2025.items())[:10]):
                    print(f"  {date}: {name}")
        
        # Save updated dataset
        output_file = 'Data/toronto_unified_hotel_analysis_with_holidays.csv'
        df_with_holidays.to_csv(output_file, index=False)
        print(f"\n✅ Updated dataset saved to: {output_file}")
        print(f"📊 Final dataset: {len(df_with_holidays)} rows, {len(df_with_holidays.columns)} columns")
        
        return df_with_holidays
    else:
        print("❌ Holiday function test failed")
        return None

if __name__ == "__main__":
    test_holiday_function()